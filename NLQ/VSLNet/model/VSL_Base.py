import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.VSLNet import VSLNet, build_optimizer_and_scheduler
from utils.data_gen import gen_or_load_dataset
from utils.data_loader import get_train_loader, get_test_loader
from utils.data_util import load_json, load_video_features, save_json
from utils.runner_utils import (
    convert_length_to_mask,
    eval_test,
    filter_checkpoints,
    get_last_checkpoint,
    set_th_config,
)


class VSLBase:
    def __init__(self, configs):
        self.configs = configs
        self.device = None
        self.dataset = None
        self.visual_features = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.model_dir = None
        self.home_dir = None

        self._setup_environment()
        self._load_data()
        self._init_model()

    def _setup_environment(self):
        set_th_config(self.configs.seed)

        cuda_str = "cuda" if self.configs.gpu_idx is None else f"cuda:{self.configs.gpu_idx}"
        self.device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # Model directory
        self.home_dir = os.path.join(
            self.configs.model_dir,
            "_".join([
                self.configs.model_name,
                self.configs.task,
                self.configs.fv,
                str(self.configs.max_pos_len),
                self.configs.predictor,
            ])
        )
        if self.configs.suffix is not None:
            self.home_dir += "_" + self.configs.suffix

        self.model_dir = os.path.join(self.home_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Tensorboard
        if self.configs.log_to_tensorboard:
            log_dir = os.path.join(self.configs.tb_log_dir, self.configs.log_to_tensorboard)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[INFO] Writing to TensorBoard at {log_dir}")

    def _load_data(self):
        self.dataset = gen_or_load_dataset(self.configs)
        self.configs.char_size = self.dataset.get("n_chars", -1)
        self.configs.word_size = self.dataset.get("n_words", -1)

        # Load features
        self.visual_features = load_video_features(
            os.path.join("data", "features", self.configs.task, self.configs.fv),
            self.configs.max_pos_len
        )
        if self.configs.video_agnostic:
            self.visual_features = {
                key: np.random.rand(*val.shape) for key, val in self.visual_features.items()
            }

        self.train_loader = get_train_loader(self.dataset["train_set"], self.visual_features, self.configs)
        self.val_loader = None if self.dataset["val_set"] is None else get_test_loader(self.dataset["val_set"], self.visual_features, self.configs)
        self.test_loader = get_test_loader(self.dataset["test_set"], self.visual_features, self.configs)

        self.configs.num_train_steps = len(self.train_loader) * self.configs.epochs
        self.num_train_batches = len(self.train_loader)

    def _init_model(self):
        self.model = VSLNet(self.configs, word_vectors=self.dataset.get("word_vector", None)).to(self.device)
        if self.configs.mode == "train":
            self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.model, self.configs)

    def train(self):
        save_json(vars(self.configs), os.path.join(self.model_dir, "configs.json"), sort_keys=True, save_pretty=True)
        best_metric = -1.0
        score_writer = open(os.path.join(self.model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
        eval_period = self.num_train_batches // 2
        global_step = 0

        for epoch in range(self.configs.epochs):
            self.model.train()
            for data in tqdm(self.train_loader, total=self.num_train_batches, desc=f"Epoch {epoch + 1}/{self.configs.epochs}"):
                global_step += 1
                (_, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels) = data

                vfeats, vfeat_lens = vfeats.to(self.device), vfeat_lens.to(self.device)
                s_labels, e_labels, h_labels = s_labels.to(self.device), e_labels.to(self.device), h_labels.to(self.device)

                if self.configs.predictor == "bert":
                    word_ids = {k: v.to(self.device) for k, v in word_ids.items()}
                    query_mask = (word_ids["input_ids"] != 0).float()
                else:
                    word_ids, char_ids = word_ids.to(self.device), char_ids.to(self.device)
                    query_mask = (word_ids != 0).float()

                video_mask = convert_length_to_mask(vfeat_lens).to(self.device)

                h_score, start_logits, end_logits = self.model(word_ids, char_ids, vfeats, video_mask, query_mask)

                highlight_loss = self.model.compute_highlight_loss(h_score, h_labels, video_mask)
                loc_loss = self.model.compute_loss(start_logits, end_logits, s_labels, e_labels)
                total_loss = loc_loss + self.configs.highlight_lambda * highlight_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.clip_norm)
                self.optimizer.step()
                self.scheduler.step()

                if self.writer and global_step % self.configs.tb_log_freq == 0:
                    self.writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                    self.writer.add_scalar("Loss/Loc", loc_loss.item(), global_step)
                    self.writer.add_scalar("Loss/Highlight", highlight_loss.item(), global_step)
                    self.writer.add_scalar("Loss/Highlight (*lambda)", (self.configs.highlight_lambda * highlight_loss.item()), global_step)
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], global_step)

                if global_step % eval_period == 0 or global_step % self.num_train_batches == 0:
                    self.model.eval()
                    result_path = os.path.join(self.model_dir, f"{self.configs.model_name}_{epoch}_{global_step}_preds.json")
                    results, mIoU, (score_str, score_dict) = eval_test(
                        model=self.model,
                        data_loader=self.val_loader,
                        device=self.device,
                        mode="val",
                        epoch=epoch + 1,
                        global_step=global_step,
                        gt_json_path=self.configs.eval_gt_json,
                        result_save_path=result_path,
                    )
                    print(score_str)
                    score_writer.write(score_str)
                    score_writer.flush()

                    if self.writer:
                        for name, value in score_dict.items():
                            self.writer.add_scalar(f"Val/{name.strip()}", value, global_step)

                    if results[0][0] >= best_metric:
                        best_metric = results[0][0]
                        ckpt_path = os.path.join(self.model_dir, f"{self.configs.model_name}_{global_step}.t7")
                        torch.save(self.model.state_dict(), ckpt_path)
                        filter_checkpoints(self.model_dir, suffix="t7", max_to_keep=3)
                    self.model.train()

        score_writer.close()

    def test(self):
        if not os.path.exists(self.model_dir):
            raise ValueError("No pre-trained weights exist")

        pre_configs = load_json(os.path.join(self.model_dir, "configs.json"))
        for k, v in pre_configs.items():
            setattr(self.configs, k, v)

        filename = get_last_checkpoint(self.model_dir, suffix="t7")
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        result_save_path = filename.replace(".t7", "_test_result.json")
        results, mIoU, score_str = eval_test(
            model=self.model,
            data_loader=self.test_loader,
            device=self.device,
            mode="test",
            result_save_path=result_save_path,
        )
        print(score_str)

    def run(self):
        if self.configs.mode == "train":
            self.train()
        elif self.configs.mode == "test":
            self.test()
