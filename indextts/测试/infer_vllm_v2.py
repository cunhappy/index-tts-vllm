import os
import random
import re
import time
import traceback
from typing import List
import uuid

import librosa
import torch
import torchaudio
# from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor
from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM
import safetensors
from loguru import logger

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model_vllm_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

import torch.nn.functional as F

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
import multiprocessing


def _create_gpt_engine_in_process(model_dir, cfg, gpu_memory_utilization, result_queue):
    """在独立进程中创建 GPT vLLM 引擎（使用 GPU 0）"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    vllm_dir = os.path.join(model_dir, "gpt")
    engine_args = AsyncEngineArgs(
        model=vllm_dir,
        tensor_parallel_size=1,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    result_queue.put(("gpt_engine", engine))


def _create_qwen_engine_in_process(model_dir, gpu_memory_utilization, result_queue):
    """在独立进程中创建 Qwen vLLM 引擎（使用 GPU 1）"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tensor_parallel_size=1,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    result_queue.put(("qwen_engine", engine))


class IndexTTS2:
    def __init__(
        self, model_dir="checkpoints", is_fp16=False, device=None, use_cuda_kernel=None, 
        gpu_memory_utilization=0.8, qwenemo_gpu_memory_utilization=0.8,
        gpt_device="cuda:0", audio_device="cuda:1"
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel.
            gpt_device (str): GPU for GPT vLLM engine (default: cuda:0)
            audio_device (str): GPU for audio processing modules (default: cuda:1)
        """
        # 检查可用 GPU 数量
        self.num_gpus = torch.cuda.device_count()
        logger.info(f">> Available GPUs: {self.num_gpus}")
        
        # 设置设备
        if self.num_gpus >= 2:
            self.gpt_device = gpt_device  # GPU 0 for GPT vLLM
            self.audio_device = audio_device  # GPU 1 for audio modules and Qwen
            logger.info(f">> Dual GPU mode: GPT on {self.gpt_device}, Audio/Qwen on {self.audio_device}")
        else:
            # 单卡模式
            self.gpt_device = "cuda:0"
            self.audio_device = "cuda:0"
            logger.info(">> Single GPU mode: all models on cuda:0")
        
        # 设置主设备（用于音频处理）
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = self.audio_device  # 音频处理使用 audio_device
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            logger.info(">> Be patient, it may take a while to run in CPU mode.")

        cfg_path = os.path.join(model_dir, "config.yaml")
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # ============ 创建 GPT vLLM 引擎 (GPU 0) ============
        logger.info(f">> Creating GPT vLLM engine on GPU 0...")
        
        # 保存当前环境变量
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        # 设置只使用 GPU 0 来创建 GPT 引擎
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        vllm_dir = os.path.join(model_dir, "gpt")
        gpt_engine_args = AsyncEngineArgs(
            model=vllm_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
        )
        indextts_vllm = AsyncLLM.from_engine_args(gpt_engine_args)
        
        # ============ 创建 Qwen vLLM 引擎 (GPU 1) ============
        logger.info(f">> Creating Qwen vLLM engine on GPU 1...")
        
        # 设置只使用 GPU 1 来创建 Qwen 引擎
        if self.num_gpus >= 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        # 单卡模式下保持 GPU 0
        
        self.qwen_emo = QwenEmotion(
            os.path.join(self.model_dir, self.cfg.qwen_emo_path),
            gpu_memory_utilization=qwenemo_gpu_memory_utilization,
        )
        
        # 恢复环境变量，让其他模块可以看到所有 GPU
        if original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        # ============ 初始化 GPT 模型 ============
        self.gpt = UnifiedVoice(indextts_vllm, **self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.gpt_device)  # GPT 相关放 GPU 0
        self.gpt.eval()
        logger.info(f">> GPT weights restored from: {self.gpt_path}")

        # ============ 预加载 CUDA kernel ============
        if self.use_cuda_kernel:
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load
                anti_alias_activation_cuda = load.load()
                logger.info(f">> Preload custom CUDA kernel for BigVGAN {anti_alias_activation_cuda}")
            except Exception as ex:
                traceback.print_exc()
                logger.info(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False

        # ============ 以下音频处理模块放在 audio_device (GPU 1) ============
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat),
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )
        self.semantic_model = self.semantic_model.to(self.audio_device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.audio_device)
        self.semantic_std = self.semantic_std.to(self.audio_device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = os.path.join(self.model_dir, "semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.audio_device)
        self.semantic_codec.eval()
        logger.info('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.audio_device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()
        logger.info(f">> s2mel weights restored from: {s2mel_path}")

        # load campplus_model
        campplus_ckpt_path = os.path.join(self.model_dir, "campplus/campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.audio_device)
        self.campplus_model.eval()
        logger.info(f">> campplus_model weights restored from: {campplus_ckpt_path}")

        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(os.path.join(self.model_dir, "bigvgan"))
        self.bigvgan = self.bigvgan.to(self.audio_device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        logger.info(f">> bigvgan weights restored from: {bigvgan_name}")

        self.bpe_path = os.path.join(self.model_dir, "bpe.model")
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        logger.info(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        logger.info(f">> bpe model loaded from: {self.bpe_path}")

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.audio_device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.audio_device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        self.speaker_dict = {}
        
        logger.info(f">> Model initialization complete!")
        logger.info(f">> GPT device: {self.gpt_device}, Audio device: {self.audio_device}")

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between sentences.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list
    
    async def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_sentence=120, **generation_kwargs):
        logger.info(">> start inference...")
        start_time = time.perf_counter()

        if use_emo_text:
            emo_audio_prompt = None
            emo_alpha = 1.0
            if emo_text is None:
                emo_text = text
            emo_dict, content = await self.qwen_emo.inference(emo_text)
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            emo_audio_prompt = None
            emo_alpha = 1.0

        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        audio, sr = librosa.load(spk_audio_prompt)
        audio = torch.tensor(audio).unsqueeze(0)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        input_features = input_features.to(self.audio_device)
        attention_mask = attention_mask.to(self.audio_device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(self.audio_device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(self.audio_device),
                                                    num_mel_bins=80,
                                                    dither=0,
                                                    sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))

        prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                    ylens=ref_target_lengths,
                                                                    n_quantizers=3,
                                                                    f0=None)[0]

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector).to(self.audio_device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        emo_audio, _ = librosa.load(emo_audio_prompt, sr=16000)
        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"]
        emo_attention_mask = emo_inputs["attention_mask"]
        emo_input_features = emo_input_features.to(self.audio_device)
        emo_attention_mask = emo_attention_mask.to(self.audio_device)
        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")

        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        
        # 需要将数据在 GPU 间传输
        # spk_cond_emb 和 emo_cond_emb 在 audio_device 上，需要传到 gpt_device
        spk_cond_emb_gpt = spk_cond_emb.to(self.gpt_device)
        emo_cond_emb_gpt = emo_cond_emb.to(self.gpt_device)
        
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.gpt_device).unsqueeze(0)

            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                emovec = self.gpt.merge_emovec(
                    spk_cond_emb_gpt,
                    emo_cond_emb_gpt,
                    torch.tensor([spk_cond_emb_gpt.shape[-1]], device=self.gpt_device),
                    torch.tensor([emo_cond_emb_gpt.shape[-1]], device=self.gpt_device),
                    alpha=emo_alpha
                )

                if emo_vector is not None:
                    emovec_mat_gpt = emovec_mat.to(self.gpt_device)
                    emovec = emovec_mat_gpt + (1 - torch.sum(weight_vector.to(self.gpt_device))) * emovec

                codes, speech_conditioning_latent = await self.gpt.inference_speech(
                    spk_cond_emb_gpt,
                    text_tokens,
                    emo_cond_emb_gpt,
                    cond_lengths=torch.tensor([spk_cond_emb_gpt.shape[-1]], device=self.gpt_device),
                    emo_cond_lengths=torch.tensor([emo_cond_emb_gpt.shape[-1]], device=self.gpt_device),
                    emo_vec=emovec,
                )
                gpt_gen_time += time.perf_counter() - m_start_time

                code_lens = []
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                        code_len = len_ - 1
                    code_lens.append(code_len)
                codes = codes[:, :code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.gpt_device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb_gpt.size(0)).to(self.gpt_device).long()
                
                latent = self.gpt(
                    speech_conditioning_latent,
                    text_tokens,
                    torch.tensor([text_tokens.shape[-1]], device=self.gpt_device),
                    codes,
                    torch.tensor([codes.shape[-1]], device=self.gpt_device),
                    emo_cond_emb_gpt,
                    cond_mel_lengths=torch.tensor([spk_cond_emb_gpt.shape[-1]], device=self.gpt_device),
                    emo_cond_mel_lengths=torch.tensor([emo_cond_emb_gpt.shape[-1]], device=self.gpt_device),
                    emo_vec=emovec,
                    use_speed=use_speed,
                )
                gpt_forward_time += time.perf_counter() - m_start_time

                # 将 GPT 输出传回 audio_device 进行后续处理
                latent = latent.to(self.audio_device)
                codes = codes.to(self.audio_device)
                code_lens = code_lens.to(self.audio_device)

                dtype = None
                with torch.amp.autocast(self.audio_device.split(':')[0], enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    latent = self.s2mel.models['gpt_layer'](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent
                    target_lengths = (code_lens * 1.72).long()

                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                   torch.LongTensor([cat_condition.size(1)]).to(self.audio_device),
                                                                   ref_mel, style, None, diffusion_steps,
                                                                   inference_cfg_rate=inference_cfg_rate)
                    vc_target = vc_target[:, :, ref_mel.size(-1):]
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                wavs.append(wav.cpu())
        end_time = time.perf_counter()

        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        logger.info(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        logger.info(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        logger.info(f">> s2mel_time: {s2mel_time:.2f} seconds")
        logger.info(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        logger.info(f">> Total inference time: {end_time - start_time:.2f} seconds")
        logger.info(f">> Generated audio length: {wav_length:.2f} seconds")
        logger.info(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        wav = wav.cpu()
        if output_path:
            if os.path.isfile(output_path):
                os.remove(output_path)
                logger.info(f">> remove old wav file: {output_path}")
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            logger.info(f">> wav file saved to: {output_path}")
            return output_path
        else:
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index


class QwenEmotion:
    def __init__(self, model_dir, gpu_memory_utilization=0.8):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        engine_args = AsyncEngineArgs(
            model=model_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
        )
        self.model = AsyncLLM.from_engine_args(engine_args)

        self.prompt = "文本情感分类"
        self.convert_dict = {
            "愤怒": "angry",
            "高兴": "happy",
            "恐惧": "fear",
            "反感": "hate",
            "悲伤": "sad",
            "低落": "low",
            "惊讶": "surprise",
            "自然": "neutral",
        }
        self.backup_dict = {"happy": 0, "angry": 0, "sad": 0, "fear": 0, "hate": 0, "low": 0, "surprise": 0,
                            "neutral": 1.0}
        self.max_score = 1.2
        self.min_score = 0.0

    def convert(self, content):
        content = content.replace("\n", " ")
        content = content.replace(" ", "")
        content = content.replace("{", "")
        content = content.replace("}", "")
        content = content.replace('"', "")
        parts = content.strip().split(',')
        parts_dict = {}
        desired_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        for part in parts:
            key_value = part.strip().split(':')
            if len(key_value) == 2:
                parts_dict[key_value[0].strip()] = part
        ordered_parts = [parts_dict[key] for key in desired_order if key in parts_dict]
        parts = ordered_parts
        if len(parts) != len(self.convert_dict):
            return self.backup_dict

        emotion_dict = {}
        for part in parts:
            key_value = part.strip().split(':')
            if len(key_value) == 2:
                try:
                    key = self.convert_dict[key_value[0].strip()]
                    value = float(key_value[1].strip())
                    value = max(self.min_score, min(self.max_score, value))
                    emotion_dict[key] = value
                except Exception:
                    continue

        for key in self.backup_dict:
            if key not in emotion_dict:
                emotion_dict[key] = 0.0

        if sum(emotion_dict.values()) <= 0:
            return self.backup_dict

        return emotion_dict

    async def inference(self, text_input):
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(text)["input_ids"]

        sampling_params = SamplingParams(
            max_tokens=2048,
        )
        tokens_prompt = TokensPrompt(prompt_token_ids=model_inputs)
        output_generator = self.model.generate(tokens_prompt, sampling_params=sampling_params, request_id=uuid.uuid4().hex)
        async for output in output_generator:
            pass
        output_ids = output.outputs[0].token_ids[:-2]

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        emotion_dict = self.convert(content)
        return emotion_dict, content
