import torch
import torch.nn as nn
from others.my_utils import l2_norm
from torch.autograd import Variable

class ALETemp(nn.Module):
	def __init__(self, params):
		super(ALETemp, self).__init__()
		if params.temp_type == 'bidirLSTM':
			self.fc_layer = nn.Linear(4*params.hidden, 768)
		else :
			self.fc_layer = nn.Linear(2*params.hidden, 768)
		self.params = params

		nn.init.xavier_uniform_(self.fc_layer.weight)
		if params.temp_type == 'LSTM':
			self.lstm_vid = nn.LSTM(1024, params.hidden, params.n_layer, batch_first=True)
			self.lstm_hand = nn.LSTM(1024, params.hidden, params.n_layer, batch_first=True)
		elif params.temp_type == 'bidirLSTM':
			self.bidirlstm_vid = nn.LSTM(1024, params.hidden, params.n_layer, batch_first=True, bidirectional=True)
			self.bidirlstm_hand = nn.LSTM(1024, params.hidden, params.n_layer, batch_first=True, bidirectional=True)
		elif params.temp_type == 'GRU':
			self.gru_vid = nn.GRU(1024, params.hidden, params.n_layer, batch_first=True)
			self.gru_hand = nn.GRU(1024, params.hidden, params.n_layer, batch_first=True)


	def forward(self, vid_features, hand_features, y):

		#print "input shape = ", vid_features.shape, hand_features.shape
		if self.params.temp_type == 'LSTM':
			avg_pool = torch.nn.AvgPool2d((7, 1), (1, 1))

			vid_features_h = avg_pool(vid_features)
			hand_features_h = avg_pool(hand_features)
			vid_features_h = vid_features_h.transpose(0, 1)
			hand_features_h = hand_features_h.transpose(0, 1)
			x_vid, (hn_vid, cn) = self.lstm_vid(vid_features, (vid_features_h, vid_features_h))#, (h0_vid, c0_vid))
			x_hand, (hn_hand, cn) = self.lstm_hand(hand_features, (hand_features_h, hand_features_h))#, (h0_hand, c0_hand))


			if self.params.n_layer > 1 :
				hn_vid = x_vid[:, -1, :]
				hn_hand = x_hand[:, -1, :]
		elif self.params.temp_type == 'GRU':

			avg_pool = torch.nn.AvgPool2d((7, 1), (1, 1))
			vid_features_h = avg_pool(vid_features)
			hand_features_h = avg_pool(hand_features)

			vid_features_h = vid_features_h.transpose(0, 1)
			hand_features_h = hand_features_h.transpose(0, 1)
			#print "hidden initial vid features = ", vid_features_h.shape
			#print "hidden initial hand features = ", hand_features_h.shape

			x_vid, hn_vid = self.gru_vid(vid_features, vid_features_h)
			x_hand, hn_hand = self.gru_hand(hand_features, hand_features_h)
			#print hn_vid.shape
			#print hn_hand.shape

		elif self.params.temp_type == 'bidirLSTM':
			avg_pool = torch.nn.AvgPool2d((7, 1), (1, 1))
			vid_features_h = avg_pool(vid_features)
			hand_features_h = avg_pool(hand_features)
			vid_features_h = vid_features_h.transpose(0, 1)
			hand_features_h = hand_features_h.transpose(0, 1)


			vid_features_h = torch.cat((vid_features_h, vid_features_h), dim=0)
			hand_features_h = torch.cat((hand_features_h, hand_features_h), dim=0)

			x_vid, _ = self.bidirlstm_vid(vid_features, (vid_features_h, vid_features_h))
			x_hand, _ = self.bidirlstm_hand(hand_features, (hand_features_h, hand_features_h))

			hn_vid = x_vid[:, -1, :]
			hn_hand = x_hand[:, -1, :]
		if self.params.temp_type == 'bidirLSTM':
			hn_vid = hn_vid.view(-1, 2*self.params.hidden)
			hn_hand = hn_hand.view(-1, 2*self.params.hidden)
		else:
			hn_vid = hn_vid.view(-1, self.params.hidden)
			hn_hand = hn_hand.view(-1, self.params.hidden)

		xh = torch.cat((hn_vid, hn_hand), 1)

		out = self.fc_layer(xh)
		if self.params.l2_norm_bool:
			y = l2_norm(y)
		y_t = y.transpose(0, 1)
		out = torch.mm(out , y_t)

		return out

def CEL_loss_fn(outputs, labels):
    loss = nn.CrossEntropyLoss()
    return loss(outputs, labels)
