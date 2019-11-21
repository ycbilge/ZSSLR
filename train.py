import time
import os
import argparse
import logging
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
from model.ALE_with_Temp import ALETemp, CEL_loss_fn
from model.ASLLVDLoader import fetch_dataloader, read_sign_class_desc_feature, train_sign_list, val_sign_list, test_sign_list, get_train_val_test_descriptions
from others.my_utils import Params, set_logger, save_checkpoint, save_dict_to_json, plot_chart, plot_confusionmatrix, calculate_accuracy
from evaluate import evaluate_model


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir_vid_stride4_5c_w8_no_avg', default='./features/5c_stride8_notemp_avg/vid/', help="Directory containing the dataset")
parser.add_argument('--data_dir_hand_stride4_5c_w8_no_avg', default='./features/5c_stride8_notemp_avg/hand/', help="Directory containing the dataset")
parser.add_argument('--experiments_dir', default='./experiments/', help="Directory containing params.json")
parser.add_argument('--textual_features_dir', default='./features/textual_features_bert_base_uncased/outputs_base/', help="Directory containing the dataset")

def train(train_dl, params, ale_net, optimizer, train_sign_descs):
	ale_net.train()
	epoch_loss_array = []
	epoch_predictions_arr = []
	epoch_labels_arr = []
	train_sign_descs = torch.from_numpy(train_sign_descs).float()
	train_sign_descs = Variable(train_sign_descs.view(-1, 768)).cuda()
	for i, sample in enumerate(train_dl):
		video_features = sample['vid_feature']
		hand_features = sample['hand_feature']
		vid_label_index = sample['vid_label_index']
		video_features = Variable(video_features).cuda()
		hand_features = Variable(hand_features).cuda()
		vid_label_index = Variable(vid_label_index).cuda()

		output_batch = ale_net(video_features, hand_features, train_sign_descs)


		loss = CEL_loss_fn(output_batch, vid_label_index)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss_array.append(loss.data.item())
		preds = np.argmax(output_batch.data.cpu().numpy(), axis=1)
		epoch_predictions_arr.extend(preds)
		epoch_labels_arr.extend(vid_label_index.data.cpu().numpy())

	epoch_accuracy, conf_mat = calculate_accuracy(epoch_predictions_arr, epoch_labels_arr)
	avg_epoch_loss = sum(epoch_loss_array) / float(len(epoch_loss_array))
	return avg_epoch_loss, epoch_accuracy


def train_and_evaluate(train_dl, val_dl, test_dl, params, ale_net, optimizer, experiments_dir, textual_features_dir):
	train_sign_descs, val_sign_descs, test_sign_descs = get_train_val_test_descriptions(textual_features_dir)
	logging.info("train desc shape = " + str(train_sign_descs.shape) + " val desc shape = " + str(val_sign_descs.shape) + " test desc shape = " + str(test_sign_descs.shape))
	acc_and_loss_plot_path = os.path.join(experiments_dir, "acc_loss.png")

	epoch_train_acc, epoch_train_loss, epoch_val_acc, epoch_val_loss = [], [], [], []
	best_val_acc = 0.0
	patience = 30
	current_score = 0
	best_val_acc_so_far = 0.0
	best_test_acc_so_far = 0.0
	for epoch in range(params.num_epochs):
		logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
		train_loss_val, train_acc = train(train_dl, params, ale_net, optimizer, train_sign_descs)
		val_loss_val, val_acc, val_conf_mat = evaluate_model(ale_net, val_dl, val_sign_descs)
		test_loss_val, test_acc, test_conf_mat = evaluate_model(ale_net, test_dl, test_sign_descs)
		epoch_train_acc.append(train_acc)
		epoch_train_loss.append(train_loss_val)
		epoch_val_loss.append(val_loss_val)
		epoch_val_acc.append(val_acc)

		logging.info(" - Train metrics : loss = " + str(train_loss_val) + " ; accuracy = " + str(train_acc))
		logging.info(" - Val metrics : loss = " + str(val_loss_val) + " ; accuracy = " + str(val_acc))
		logging.info(" - Test metrics : loss = " + str(test_loss_val) + " ; accuracy = " + str(test_acc))
		is_best = val_acc >= best_val_acc
		save_checkpoint({'epoch': epoch + 1, 'state_dict': ale_net.state_dict(),
		                 'optim_dict': optimizer.state_dict()},
		                is_best=is_best, checkpoint=experiments_dir)

		if is_best:
			logging.info("best val acc = " + str(best_val_acc))
			logging.info("- Found new best accuracy")
			best_val_acc = val_acc
			val_best_confmat_path = os.path.join(experiments_dir, "val_best_confmat.png")
			test_best_confmat_path = os.path.join(experiments_dir, "test_best_confmat.png")
			best_json_path = os.path.join(experiments_dir, "metrics_val_best_weights.json")
			plot_confusionmatrix(val_conf_mat, val_sign_list, val_best_confmat_path)
			plot_confusionmatrix(test_conf_mat, test_sign_list, test_best_confmat_path)
			save_dict_to_json(best_val_acc, test_acc, best_json_path)

		if epoch == 0:
			best_val_acc_so_far = val_acc
			best_test_acc_so_far = test_acc
		else :
			if val_acc <= best_val_acc_so_far:
				current_score += 1
				logging.info('Best Val accuracy not improved patience is ' + str(current_score) + ' out of ' + str(patience))
			else:
				best_val_acc_so_far = val_acc
				best_test_acc_so_far = test_acc
				current_score = 0
		logging.info("best val acc = " + str(best_val_acc) + " - best test acc; so far = " + str(best_test_acc_so_far))
		if current_score == patience:
			break




	plot_chart(epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc,"Resulting Plot, Acc and Loss Function", acc_and_loss_plot_path)





def get_model(params):
    ale_net = ALETemp(params).cuda()
    optimizer = optim.Adam(ale_net.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return ale_net, optimizer, scheduler



def main():
	start = time.time()
	args = parser.parse_args()
	json_path = os.path.join(args.experiments_dir, 'params.json')
	assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
	params = Params(json_path)
	set_logger(os.path.join(args.experiments_dir, 'train.log'))

	logging.info("Feature vid location = " + args.data_dir_vid_stride4_5c_w8_no_avg)
	logging.info("Feature hand location = " + args.data_dir_hand_stride4_5c_w8_no_avg)
	dataloaders = fetch_dataloader(['train', 'val', 'test'], params, args.data_dir_vid_stride4_5c_w8_no_avg, args.data_dir_hand_stride4_5c_w8_no_avg)
	train_dl, val_dl, test_dl = dataloaders['train'], dataloaders['val'], dataloaders['test']

	logging.info("Loading the datasets..." + " Training length = " + str(len(train_dl)) + " Val length = " + str(len(val_dl)) + " Test length = " + str(len(test_dl)))

	ale_net, optimizer, scheduler = get_model(params)
	logging.info(str(ale_net) + " - " + str(optimizer))
	train_and_evaluate(train_dl, val_dl, test_dl, params, ale_net, optimizer, args.experiments_dir, args.textual_features_dir)
	end = time.time()
	elapsed_time = end - start
	str_time = str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
	logging.info("Training finished for {} epoch(s) in {}".format(params.num_epochs, str_time))


if __name__== '__main__':
    main()
