import os
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model.VSLNet_t7 import VSLNet, build_optimizer_and_scheduler
from util.data_gen import gen_or_load_dataset, gen_or_load_eval_dataset
from util.data_loader_t7 import get_train_loader, get_test_loader
from util.data_util import load_video_features, save_json, load_json
from util.runner_utils_t7 import set_th_config, convert_length_to_mask, eval_test, filter_checkpoints, \
    get_best_checkpoint

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='data/dataset/medvidqa/processed_data', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='medvidqa', help='target task')
parser.add_argument('--test_data_file_name', type=str, default='test.json', help='target task')
parser.add_argument('--fv', type=str, default='new', help='[new | org] for visual features')
parser.add_argument('--max_pos_len', type=int, default=800, help='maximal position sequence length allowed')
# model parameters
parser.add_argument("--word_size", type=int, default=None, help="number of words")
parser.add_argument("--char_size", type=int, default=None, help="number of characters")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension")
parser.add_argument("--char_dim", type=int, default=50, help="character dimension, set to 100 for activitynet")
parser.add_argument("--dim", type=int, default=128, help="hidden size")
parser.add_argument("--highlight_lambda", type=float, default=1.0, help="lambda for highlight region")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
parser.add_argument('--predictor', type=str, default='rnn', help='[rnn | transformer]')
# training/evaluation parameters
parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
parser.add_argument("--seed", type=int, default=12345, help="random seed")
parser.add_argument("--mode", type=str, default="train", help="[train | test]")

parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--extend", type=float, default=0.25, help="highlight region extension")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument('--model_dir', type=str, default='ckpt_t7_medvidqa_final', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='vslnet', help='model name')
parser.add_argument('--suffix', type=str, default=None, help='')

configs = parser.parse_args()

# set pytorch configs
set_th_config(configs.seed)


# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# create model dir
if configs.highlight_lambda>0:
    configs.model_name = 'vslnet'
else:
    configs.model_name = 'vslbase'
home_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name, configs.task,
                                                     str(configs.max_pos_len), 'extend'+'_'+str(configs.extend), configs.predictor]))
if configs.suffix is not None:
    home_dir = home_dir + '_' + configs.suffix
model_dir = os.path.join(home_dir, "model")

print(configs)
# train and test
if configs.mode.lower() == 'train':

    experiment_name ='medvidqa'+'-'+ 'fps-16' +'-'+ 'stride-16' +'-'+ str(configs.max_pos_len)+'-'+ \
                        configs.predictor+'-'+'extend'+'-'+str(configs.extend)+'-'+'lr'+'-'+ str(configs.init_lr)+'-'+configs.model_name

    # prepare or load dataset
    dataset = gen_or_load_dataset(configs)
    configs.char_size = dataset['n_chars']
    configs.word_size = dataset['n_words']

    # get train and test loader
    visual_features = load_video_features(os.path.join('data', 'features', configs.task), configs.max_pos_len)
    train_loader = get_train_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs)
    val_loader = get_test_loader(dataset=dataset['val_set'], video_features=visual_features, configs=configs)
    configs.num_train_steps = len(train_loader) * configs.epochs
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_period = num_train_batches // 2
    save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    # build model
    model = VSLNet(configs=configs, word_vectors=dataset['word_vector']).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    # start training
    best_r1i7 = -1.0
    score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
    print('start training...', flush=True)
    global_step = 0
    for epoch in range(configs.epochs):
        model.train()
        for data in tqdm(train_loader, total=num_train_batches, desc='Epoch %3d / %3d' % (epoch + 1, configs.epochs)):
            global_step += 1
            _, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels = data
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            s_labels, e_labels, h_labels = s_labels.to(device), e_labels.to(device), h_labels.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            # compute logits
            h_score, start_logits, end_logits = model(word_ids, char_ids, vfeats, video_mask, query_mask)
            # compute loss
            highlight_loss = model.compute_highlight_loss(h_score, h_labels, video_mask)
            loc_loss = model.compute_loss(start_logits, end_logits, s_labels, e_labels)
            total_loss = loc_loss + configs.highlight_lambda * highlight_loss
            # compute and apply gradients
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()
            ## print loss
            if global_step % num_train_batches == 0:
                print('\nEpoch: %2d | Step: %5d | total_loss: %.2f | loc_loss: %.2f | highlight_loss: %.2f' % (
                epoch + 1, global_step, total_loss, loc_loss, highlight_loss), flush=True)

            # evaluate
            if global_step % eval_period == 0 or global_step % num_train_batches == 0:
                model.eval()

                #### evalauting on val set
                r1i3, r1i5, r1i7, mi, score_str = eval_test(model=model, data_loader=val_loader, device=device,
                                                            mode='val', epoch=epoch + 1, global_step=global_step)
                print('\nEpoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % (
                    epoch + 1, global_step, r1i3, r1i5, r1i7, mi), flush=True)
                score_writer.write(score_str)
                score_writer.flush()



                if r1i7 > best_r1i7:
                    best_r1i7 = r1i7
                    torch.save(model.state_dict(), os.path.join(model_dir, '{}_{}_{}.t7'.format(configs.model_name
                                                                                             , global_step, str(r1i7))))

                model.train()

    filter_checkpoints(model_dir, suffix='t7', max_to_keep=3)
    score_writer.close()

elif configs.mode.lower() == 'test':

    if configs.highlight_lambda > 0:
        configs.model_name = 'vslnet'
    else:
        configs.model_name = 'vslbase'
    home_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name, configs.task,
                                                         str(configs.max_pos_len), 'extend' + '_' + str(configs.extend),
                                                         configs.predictor]))
    if configs.suffix is not None:
        home_dir = home_dir + '_' + configs.suffix



    print('start testing...', flush=True)
    print(f"test file name...{configs.test_data_file_name}")
    model_dir = os.path.join(home_dir, "model")

    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')

    #### process or load dataset

    dataset = gen_or_load_eval_dataset(configs)
    configs.char_size = dataset['n_chars']
    configs.word_size = dataset['n_words']

    # get test loader
    visual_features = load_video_features(os.path.join('data', 'features', configs.task), configs.max_pos_len)
    test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs)

    num_test_batches = len(test_loader)

    # load previous configs
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # build model
    model = VSLNet(configs=configs, word_vectors=dataset['word_vector']).to(device)
    # get best_on_val checkpoint file
    filename = get_best_checkpoint(model_dir, suffix='t7')
    print(f"Loaded model {filename}")
    model.load_state_dict(torch.load(filename))
    model.eval()
    r1i3, r1i5, r1i7, mi, _ = eval_test(model=model, data_loader=test_loader, device=device,
                                                     mode='test')
    print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m", flush=True)


