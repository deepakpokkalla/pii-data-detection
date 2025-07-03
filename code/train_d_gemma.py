import json
import os
import time
from collections import OrderedDict

# import cudf.pandas
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# cudf.pandas.install()
from transformers import BitsAndBytesConfig

try:
    from d_gemma.pii_dataset import PIIDataset
    from d_gemma.pii_loader import PIIDataCollator, show_batch
    from d_gemma.pii_model import GemmaForPII
    from d_gemma.pii_optimizer import get_optimizer
    from utils.label_mapping import label2id
    from utils.metric_utils import compute_metrics
    from utils.train_utils import (AverageMeter, as_minutes,
                                   create_reference_df,
                                   get_custom_cosine_schedule_with_warmup,
                                   get_lr, setup_training_run)

except Exception as e:
    print(e)
    raise ImportError

logger = get_logger(__name__)


pd.options.display.max_colwidth = 1000

# -------- Evaluation -------------------------------------------------------------#


def to_list(t):
    return t.cpu().numpy().tolist()


def get_pred_df(documents, word_ids, predictions, id2label):
    """
    Aggregate predictions by averaging probabilities over tokens
    """
    df = pd.DataFrame({'document': documents, 'token': word_ids, 'probs': predictions})
    df = df.explode(['token', 'probs']).reset_index(drop=True)

    # filtering ---
    df = df[df['token'] != -1]
    df = df[df['probs'].apply(lambda x: x[-1] < 0.991)]  # filter out very high outside probabilities

    df = df.groupby(['document', 'token'])['probs'].agg(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()

    df['outside_prob'] = df['probs'].apply(lambda x: x[-1])
    df['most_probable_label'] = df['probs'].apply(lambda x: id2label[np.argmax(x[:-1])])

    return df


def find_best_threshold(pred_df, reference_df):
    thresholds = np.linspace(0.75, 0.99, 24)
    th_curve_points = []

    best_f5, best_threshold = -1, None
    for threshold in thresholds:
        pred_df['label'] = np.where(pred_df['outside_prob'] <= threshold, pred_df['most_probable_label'], 'O')
        oof_df = pred_df[pred_df['label'] != 'O'].copy()
        eval_dict = compute_metrics(oof_df, reference_df)
        f5 = eval_dict["lb"]
        th_curve_points.append((round(threshold, 2), round(f5, 4)))
        if f5 > best_f5:
            best_f5, best_threshold = f5, threshold

    return best_threshold, best_f5, th_curve_points


def run_evaluation(cfg, accelerator, model, valid_dl, id2label, reference_df):
    # comment
    model.eval()

    all_documents = []
    all_tokens = []
    all_probs = []

    progress_bar = tqdm(range(len(valid_dl)), disable=not accelerator.is_local_main_process)

    for batch in valid_dl:
        with torch.no_grad():
            outputs = model(**batch)
            predictions = F.softmax(outputs.logits, dim=-1)

        documents = batch["document"]
        word_ids = accelerator.pad_across_processes(batch['word_ids'], dim=1, pad_index=-1)
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-1)

        documents, word_ids, predictions = accelerator.gather_for_metrics((documents, word_ids, predictions))
        documents, word_ids, predictions = to_list(documents), to_list(word_ids), to_list(predictions)

        all_documents.extend(documents)
        all_tokens.extend(word_ids)
        all_probs.extend(predictions)
        progress_bar.update(1)
    progress_bar.close()

    # ---
    pred_df = get_pred_df(all_documents, all_tokens, all_probs, id2label)

    # determine best outside threshold ---
    best_threshold, best_f5, th_curve_points = find_best_threshold(pred_df, reference_df)

    # OOF predictions ---
    outside_threshold = cfg.outside_threshold
    pred_df['label'] = pred_df.apply(
        lambda x: x['most_probable_label'] if x['outside_prob'] <= outside_threshold else 'O', axis=1
    )

    oof_df = pred_df[pred_df['label'] != 'O'].copy()

    oof_df['row_id'] = list(range(len(oof_df)))
    oof_df = oof_df[["row_id", "document", "token", "label"]].copy()
    oof_df = oof_df.reset_index(drop=True)

    # compute metrics --
    eval_dict = compute_metrics(oof_df, reference_df)

    to_return = {
        "scores": eval_dict,
        "oof_df": oof_df,
        "best_threshold": best_threshold,
        "best_f5": best_f5,
        "th_curve_points": th_curve_points,
        "pred_df": pred_df,
    }

    return to_return


# -------- Main Function ----------------------------------------------------------------#


@hydra.main(version_base=None, config_path="../config/d_gemma", config_name="conf_d_gemma_2b")
def main(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    accelerator = setup_training_run(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit*50 + suffix)

    print_line()
    id2label = {v: k for k, v in label2id.items()}

    data_dir = cfg.input_data_dir

    with open(os.path.join(data_dir, "train_dict.json"), "r") as f:
        example_dict = json.load(f)

    # ------- Data Split ----------------------------------------------------------------#
    comp_dataset = example_dict["competition"]

    valid_examples = []
    train_examples_comp = []

    for ex in comp_dataset:
        if ex['document'] % 4 == cfg.fold:
            valid_examples.append(ex)
        else:
            train_examples_comp.append(ex)

    if cfg.all_data:
        train_examples_comp = comp_dataset

    # modes of training ----
    train_examples = []
    for _ in range(cfg.comp_multiplier):
        train_examples.extend(train_examples_comp)
    accelerator.print(f"# examples in train data (competition): {len(train_examples)}")

    # add external sets ---
    print_line()
    accelerator.print("adding external datasets...")
    external_examples = []
    assert len(cfg.datamix_sets) == len(cfg.datamix_weights)
    for set_name, weight in zip(cfg.datamix_sets, cfg.datamix_weights):
        subset = example_dict[set_name]
        if weight > 0:
            accelerator.print(f"adding {set_name} (# examples = {len(subset)}) with multiplier {weight}")
            for _ in range(weight):
                external_examples.extend(subset)
    print_line()

    if cfg.sample_external_examples:
        frac = cfg.external_sampling_rate
        n = int(frac*len(external_examples))
        external_examples = external_examples[:n]
        accelerator.print(f"sampling {n} external examples...")

    train_examples.extend(external_examples)

    accelerator.print(f"# examples in train data: {len(train_examples)}")
    accelerator.print(f"# examples in valid data: {len(valid_examples)}")

    print_line()

    # with accelerator.main_process_first():
    dataset_creator = PIIDataset(cfg, label2id)

    train_ds = dataset_creator.get_dataset(train_examples)  # [:1024]
    valid_ds = dataset_creator.get_dataset(valid_examples)

    # filter out examples with no spans
    print_line()
    accelerator.print("filtering out examples with no spans...")
    accelerator.print(f"# examples in train data: {len(train_ds)}")
    train_ds = train_ds.filter(lambda x: len(x["span_head_idxs"]) > 4)  # at least 5 tokens
    accelerator.print(f"# examples in train data after filtering: {len(train_ds)}")
    valid_ds = valid_ds.filter(lambda x: len(x["span_head_idxs"]) > 0)
    print_line()

    tokenizer = dataset_creator.tokenizer

    # create reference dataframe for evaluation --
    reference_df = create_reference_df(valid_examples)
    accelerator.print(reference_df.head(5))

    # ------- data loaders --------------------------------------------------------------#
    # sort valid dataset for faster evaluation
    valid_ds = valid_ds.sort("input_length")

    data_collector = PIIDataCollator(tokenizer=tokenizer, pad_to_multiple_of=16)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collector,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collector,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch --------------------------------------------------------------------#
    for batch_idx, b in enumerate(train_dl):
        show_batch(b, tokenizer, id2label, task='training', print_fn=accelerator.print)
        if batch_idx >= 4:
            break
    print_line()

    # ------- Config --------------------------------------------------------------------#
    accelerator.print("config for the current run:")
    accelerator.print(json.dumps(cfg_dict, indent=4))
    print_line()

    # ------- Model ---------------------------------------------------------------------#
    print_line()
    accelerator.print("creating the PII Data Detection model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["classification_head"],
    )

    num_labels = len(label2id)

    base_model = GemmaForPII.from_pretrained(
        cfg.model.backbone_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        # quantization_config=bnb_config,
    )

    base_model.config.pretraining_tp = 1

    # lora --
    peft_config = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        lora_dropout=cfg.model.lora.lora_dropout,
        bias="none",
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        target_modules=cfg_dict["model"]["lora"]["target_modules"],
        modules_to_save=cfg_dict["model"]["lora"]["modules_to_save"],
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    print_line()

    # ------- Optimizer -----------------------------------------------------------------#
    print_line()
    accelerator.print("creating the optimizer...")
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)
    # ------- Prepare -------------------------------------------------------------------#

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    ######################################################################################
    # Training
    ######################################################################################

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    accelerator.print(f"# pre-training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# pre-training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps for pre-training: {num_warmup_steps}")

    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ------- training setup ------------------------------------------------------------#
    best_lb = -1.  # higher is better
    patience_tracker = 0
    current_iteration = 0

    # ------- training  -----------------------------------------------------------------#
    start_time = time.time()
    accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        # Training ------
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())

            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # ------- evaluation  -------------------------------------------------------#
            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                # set model in eval mode
                model.eval()
                eval_response = run_evaluation(cfg, accelerator, model, valid_dl, id2label, reference_df)

                scores_dict = eval_response["scores"]
                oof_df = eval_response["oof_df"]
                pred_df = eval_response["pred_df"]
                optim_outside_threshold = round(eval_response["best_threshold"], 2)
                optim_f5 = eval_response["best_f5"]
                lb = scores_dict["lb"]

                print_line()
                et = as_minutes(time.time()-start_time)
                accelerator.print(
                    f""">>> Epoch {epoch+1} | Step {current_iteration} | Time: {et} | Training Loss: {loss_meter.avg:.4f}"""
                )
                accelerator.print(f">>> Current LB (F5) = {round(lb, 4)}")
                accelerator.print(f">>> Optim F5@th=({optim_outside_threshold}) = {round(optim_f5, 4)}")

                print_line()

                granular_scores = scores_dict["ents_per_type"]

                accelerator.print(f">>> Granular Evaluation:")
                granular_scores = OrderedDict(sorted(granular_scores.items()))

                for k, v in granular_scores.items():
                    accelerator.print(
                        f"> [{k:<24}] P: {round(v['p'], 3):<8} | R: {round(v['r'], 3):<8} | F5: {round(v['f5'], 3):<8} |"
                    )

                print_line()

                accelerator.print(f">>> Threshold Curve Points:")
                for th_i, f5_i in eval_response["th_curve_points"]:
                    accelerator.print(f"> F5@{th_i} = {f5_i}")

                print_line()

                is_best = False
                if lb >= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # -----
                    best_dict = dict()
                    for k, v in scores_dict.items():
                        best_dict[f"{k}_at_best"] = v
                else:
                    patience_tracker += 1

                if is_best:
                    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_best.csv"), index=False)
                    pred_df.to_parquet(os.path.join(cfg.outputs.model_dir, f"pred_df_best.parquet"), index=False)
                else:
                    accelerator.print(f">>> patience reached {patience_tracker}/{cfg_dict['train_params']['patience']}")
                    accelerator.print(f">>> current best score: {round(best_lb, 4)}")

                oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_last.csv"), index=False)
                pred_df.to_parquet(os.path.join(cfg.outputs.model_dir, f"pred_df_last.parquet"), index=False)

                # saving last & best checkpoints -----
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                if cfg.save_model:
                    unwrapped_model.save_pretrained(
                        f"{cfg.outputs.model_dir}/last",
                        state_dict=accelerator.get_state_dict(model),
                        save_function=accelerator.save,
                    )

                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

                # logging ----
                if cfg.use_wandb:
                    accelerator.log({"lb": lb}, step=current_iteration)
                    accelerator.log({"best_lb": best_lb}, step=current_iteration)

                    # -- log scores dict
                    accelerator.log(scores_dict, step=current_iteration)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

                # early stopping ----
                if patience_tracker >= cfg.train_params.patience:
                    print("stopping early")
                    model.eval()
                    accelerator.end_training()
                    return

    # # save the final model
    # # saving last & best checkpoints -----
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)

    # if cfg.save_model:
    #     unwrapped_model.save_pretrained(
    #         f"{cfg.outputs.model_dir}/last",
    #         state_dict=accelerator.get_state_dict(model),
    #         save_function=accelerator.save,
    #     )

    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")


if __name__ == "__main__":
    main()
