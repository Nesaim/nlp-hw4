import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import set_random_seeds
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--skip_test', action='store_true',
                        help='Train and validate only; defer hidden-test generation')
    parser.add_argument('--eval_only', action='store_true',
                        help='Load the best checkpoint without training')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Check one training update and two dev generations, without saving')
    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    import json
    with open(os.path.join(checkpoint_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        with open('metrics_history.jsonl', 'a') as history:
            history.write(json.dumps({
                'epoch': epoch, 'train_loss': float(tr_loss),
                'dev_loss': float(eval_loss), 'record_f1': float(record_f1),
                'record_em': float(record_em), 'sql_em': float(sql_em),
                'execution_error_fraction': float(error_rate)
            }) + '\n')
        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def _decode_batch(args, model, encoder_input, encoder_mask, starts, tokenizer):
    generated = model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        decoder_input_ids=starts.to(DEVICE),
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=False,
    )
    queries = tokenizer.batch_decode(
        generated, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return [q.replace("\r", " ").replace("\n", " ").strip() for q in queries]


@torch.no_grad()
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path,
               gt_record_path, model_record_path):
    model.eval()
    tokenizer = dev_loader.dataset.tokenizer
    total_loss, total_tokens = 0.0, 0
    queries = []
    for enc, mask, dec, targets, starts in tqdm(dev_loader, desc="Dev"):
        enc, mask = enc.to(DEVICE), mask.to(DEVICE)
        dec, targets = dec.to(DEVICE), targets.to(DEVICE)
        logits = model(input_ids=enc, attention_mask=mask,
                       decoder_input_ids=dec).logits
        token_count = (targets != PAD_IDX).sum().item()
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=PAD_IDX, reduction='sum'
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite development loss")
        total_loss += loss.item()
        total_tokens += token_count
        queries.extend(_decode_batch(args, model, enc, mask, starts, tokenizer))

    if len(queries) != len(dev_loader.dataset):
        raise ValueError("Dev prediction count does not match dataset")
    for path in (model_sql_path, model_record_path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    save_queries_and_records(queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(bool(msg) for msg in error_msgs) / len(queries)
    return total_loss / max(total_tokens, 1), record_f1, record_em, sql_em, error_rate


@torch.no_grad()
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    tokenizer = test_loader.dataset.tokenizer
    queries = []
    for enc, mask, starts in tqdm(test_loader, desc="Test generation"):
        queries.extend(_decode_batch(
            args, model, enc.to(DEVICE), mask.to(DEVICE), starts, tokenizer
        ))
    if len(queries) != len(test_loader.dataset):
        raise ValueError("Test prediction count does not match dataset")
    for path in (model_sql_path, model_record_path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    save_queries_and_records(queries, model_sql_path, model_record_path)
    print(f"Saved {len(queries)} ordered test predictions to {model_sql_path}")


def smoke_test(args, model, train_loader, dev_loader):
    # No checkpoints or prediction files are changed by this check.
    model.train()
    enc, mask, dec, targets, _ = next(iter(train_loader))
    enc, mask = enc[:2].to(DEVICE), mask[:2].to(DEVICE)
    dec, targets = dec[:2].to(DEVICE), targets[:2].to(DEVICE)
    assert enc.shape == mask.shape
    assert dec.shape == targets.shape
    assert torch.all(dec[:, 0] == model.config.decoder_start_token_id).item()
    non_pad = targets != PAD_IDX
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    logits = model(input_ids=enc, attention_mask=mask, decoder_input_ids=dec).logits
    loss = nn.functional.cross_entropy(logits[non_pad], targets[non_pad])
    if not torch.isfinite(loss).item():
        raise RuntimeError("Smoke-test loss is not finite")
    loss.backward()
    optimizer.step()
    print('Smoke test: training loss =', loss.item())
    print('Batch shapes:', tuple(enc.shape), tuple(dec.shape))
    model.eval()
    enc, mask, _, _, starts = next(iter(dev_loader))
    with torch.no_grad():
        queries = _decode_batch(args, model, enc[:2].to(DEVICE),
                               mask[:2].to(DEVICE), starts[:2], dev_loader.dataset.tokenizer)
    print('Dev generation examples (not a performance evaluation):', queries)
    print('Smoke test passed; no checkpoint was saved.')


def main():
    # Get key arguments
    args = get_args()
    set_random_seeds(args.seed)
    print("Device:", DEVICE, "Seed:", args.seed)
    if args.use_wandb and wandb is None:
        raise ImportError("Install wandb to use --use_wandb")
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = (load_model_from_checkpoint(args, best=True)
             if args.eval_only else initialize_model(args))
    if args.smoke_test:
        smoke_test(args, model, train_loader, dev_loader)
        return
    if not args.eval_only and args.max_n_epochs <= 0:
        raise ValueError("Set --max_n_epochs to a positive number")
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    if not args.eval_only:
        train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    if not args.skip_test:
        test_inference(args, model, test_loader, model_sql_path, model_record_path)
    else:
        print('Test generation skipped; develop using the dev set first.')

if __name__ == "__main__":
    main()
