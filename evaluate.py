import argparse
import torch
import numpy as np
import os
from torch.autograd import Variable
from others.my_utils import calculate_accuracy, Params, load_checkpoint
from model.ALE_with_Temp import CEL_loss_fn, ALETemp
from model.ASLLVDLoader import test_sign_list
from model.ASLLVDLoader import get_train_val_test_descriptions, fetch_dataloader


parser = argparse.ArgumentParser()
parser.add_argument('--restore_experiments_location', default='./pretrained_model/batch-size-64_learning-rate_0.0009_weight_decay_0.001/', help='location of the experiments containing pretrained weights')
parser.add_argument('--data_dir_vid_stride4_5c_w8_no_avg', default='./features/5c_stride8_notemp_avg/vid/', help="Directory containing the dataset")
parser.add_argument('--data_dir_hand_stride4_5c_w8_no_avg', default='./features/5c_stride8_notemp_avg/hand/', help="Directory containing the dataset")
parser.add_argument('--textual_features_dir', default='./features/textual_features_bert_base_uncased/outputs_base/', help="Directory containing the dataset")


def get_topk_accuracy(outputs, labels, topk):
    preds_topk = np.array([output.argsort()[-topk:][::-1] for output in outputs])
    acc_dict = {}
    for label, preds in zip(labels, preds_topk):
        if label in acc_dict:
            if label in preds:
                acc_dict[label].append(1)
            else:
                acc_dict[label].append(0)
        else :
            acc_dict[label] = []
            if label in preds:
                acc_dict[label].append(1)
            else:
                acc_dict[label].append(0)
    counter = 0
    avg = 0
    for key, value in acc_dict.items():
        count_val = len(acc_dict[key])
        tp_val = acc_dict[key].count(1)
        one_avg = tp_val/float(count_val)
        avg += one_avg
        counter += 1
    return avg/float(counter)



def evaluate_model(model, dataloader, sign_descs):
    model.eval()
    all_prediction_indexes = []
    all_label_indexes = []
    epoch_loss_arr = []
    sign_descs = torch.from_numpy(sign_descs).float()
    sign_descs = Variable(sign_descs.view(-1, 768)).cuda()

    for sample in dataloader:
		video_features = sample['vid_feature']
		hand_features = sample['hand_feature']
		vid_label_index = sample['vid_label_index']
		video_features = Variable(video_features).cuda()
		hand_features = Variable(hand_features).cuda()
		vid_label_index = Variable(vid_label_index).cuda()
		output_batch = model(video_features, hand_features, sign_descs)
		loss = CEL_loss_fn(output_batch, vid_label_index)
		epoch_loss_arr.append(loss.data.item())
		preds = np.argmax(output_batch.data.cpu().numpy(), axis=1)


		all_prediction_indexes.extend(preds)
		all_label_indexes.extend(vid_label_index.data.cpu().numpy())
    epoch_accuracy, conf_mat = calculate_accuracy(all_prediction_indexes, all_label_indexes)

    avg_epoch_loss = sum(epoch_loss_arr) / float(len(epoch_loss_arr))
    return avg_epoch_loss, epoch_accuracy, conf_mat


def evaluate_and_get_topk_accuracy(model, dataloader, sign_descs):
    """Basically, evaluate the model with pretrained weights """
    model.eval()
    all_prediction_indexes = []
    all_label_indexes = []
    epoch_loss_arr = []
    sign_descs = torch.from_numpy(sign_descs).float()
    sign_descs = Variable(sign_descs.view(-1, 768)).cuda()

    pred_arrs = np.empty([])
    a = 0
    vid_label_names = []

    for sample in dataloader:
        video_features = sample['vid_feature']
        hand_features = sample['hand_feature']
        vid_label_index = sample['vid_label_index']
        vid_label_names.append(sample['vid_label'])
        video_features = Variable(video_features).cuda()
        hand_features = Variable(hand_features).cuda()
        vid_label_index = Variable(vid_label_index).cuda()


        output_batch = model(video_features, hand_features, sign_descs)


        loss = CEL_loss_fn(output_batch, vid_label_index)
        epoch_loss_arr.append(loss.data.item())
        preds = np.argmax(output_batch.data.cpu().numpy(), axis=1)
        batch_preds = output_batch.data.cpu().numpy()

        if a == 0:
            pred_arrs = np.concatenate([batch_preds])
            a += 1
        else:
            pred_arrs = np.concatenate([pred_arrs, batch_preds])


        all_prediction_indexes.extend(preds)
        all_label_indexes.extend(vid_label_index.data.cpu().numpy())

    epoch_accuracy, conf_mat = calculate_accuracy(all_prediction_indexes, all_label_indexes)
    top_1 = get_topk_accuracy(pred_arrs, all_label_indexes, 1)
    top_2 = get_topk_accuracy(pred_arrs, all_label_indexes, 2)
    top_5 = get_topk_accuracy(pred_arrs, all_label_indexes, 5)
    avg_epoch_loss = sum(epoch_loss_arr) / float(len(epoch_loss_arr))
    return avg_epoch_loss, epoch_accuracy, conf_mat, top_1, top_2, top_5


if __name__ == '__main__':

    args = parser.parse_args()
    train_sign_descs, val_sign_descs, test_sign_descs = get_train_val_test_descriptions(args.textual_features_dir)
    avg_val_top_1, avg_val_top_2, avg_val_top_5, avg_val = 0.0, 0.0, 0.0, 0.0
    avg_test_top_1, avg_test_top_2, avg_test_top_5, avg_test = 0.0, 0.0, 0.0, 0.0
    for i in range(1, 6):
        path = os.path.join(args.restore_experiments_location, str(i), 'best.pth.tar')
        json_path = os.path.join(args.restore_experiments_location, str(i), 'params.json')
        params = Params(json_path)
        dataloaders = fetch_dataloader(['train', 'val', 'test'], params, args.data_dir_vid_stride4_5c_w8_no_avg, args.data_dir_hand_stride4_5c_w8_no_avg)
        train_dl, val_dl, test_dl = dataloaders['train'], dataloaders['val'], dataloaders['test']
        model = ALETemp(params).cuda()
        model = load_checkpoint(path, model)
        val_loss_val, val_acc, val_conf_mat, val_top1, val_top2, val_top5 = evaluate_and_get_topk_accuracy(model, val_dl, val_sign_descs)
        test_loss_val, test_acc, test_conf_mat, test_top1, test_top2, test_top5 = evaluate_and_get_topk_accuracy(model, test_dl, test_sign_descs)
        avg_val_top_1 += val_top1
        avg_val_top_2 += val_top2
        avg_val_top_5 += val_top5

        avg_test_top_1 += test_top1
        avg_test_top_2 += test_top2
        avg_test_top_5 += test_top5
        avg_val += val_acc
        avg_test += test_acc

        print "---"
        print "Experiment name = ", path
        print "val acc = ", val_acc
        print "test acc = ", test_acc
        print "---"


    print "###"
    print "Overall validation results for = ", args.restore_experiments_location
    print "val acc = ", avg_val / float(5)
    print "val top-1 = ", avg_val_top_1 / float(5)
    print "val top-2 = ", avg_val_top_2 / float(5)
    print "val top-5 = ", avg_val_top_5 / float(5)
    print "Overall test results for = ", args.restore_experiments_location
    print "test acc = ", avg_test / float(5)
    print "test top-1 = ", avg_test_top_1 / float(5)
    print "test top-2 = ", avg_test_top_2 / float(5)
    print "test top-5 = ", avg_test_top_5 / float(5)
    print "###"
