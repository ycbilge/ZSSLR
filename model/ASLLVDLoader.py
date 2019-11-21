import pickle
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


train_sign_list = ["EXCUSE","INCLUDE_OR_INVOLVE", "SHELF_OR_FLOOR", "ANSWER", "BOSS", "DRESS_OR_CLOTHES", "MARRY", "STAND-UP", "DISAPPOINT", "EXPERT", "CANCEL_OR_CRITICIZE", "FED-UP_OR_FULL"
             ,"GUITAR", "EMPHASIZE", "GOVERNMENT", "LOOK", "AFRAID", "COURT", "MEDICINE", "HELLO", "CONFLICT_OR_INTERSECTION", "LESS", "OF-COURSE", "DISMISS", "DARK", "SILLY", "HOME"
             ,"BLUE", "APPOINTMENT", "DISCONNECT", "A-LOT", "ENTER", "MAD", "COLD", "DECIDE", "ARRIVE", "INFORM", "PROCEED", "MISS_OR_ASSUME", "LETTER_OR_MAIL", "KEEP", "FLY-BY-PLANE"
             ,"DEAF", "HIGH","BEAUTIFUL", "AGAIN", "HAPPEN", "DEPRESS", "EMBARRASS", "DEPOSIT", "DRUNK", "DEVELOP", "OVER_OR_AFTER", "BRAVE_OR_RECOVER", "AVOID_OR_FALL-BEHIND","FULL"
             ,"BLAME", "GOAL", "ART_OR_DESIGN", "ALLOW", "LIVE", "FUTURE","BOY", "NICE_OR_CLEAN", "DRY", "HAVE", "TAKE-UP", "HEAVY", "GROW", "EARTH", "FRIDAY", "DOWN", "BORE", "CENTER"
             ,"CHEAP", "EVERYDAY", "DIVORCE", "FORGET", "AWKWARD", "GRANDFATHER", "CRUEL", "GRADUATE", "BEER", "LEAVE-THERE", "ANY", "CHEMISTRY", "BROWN", "FRIEND", "LEFT", "FREE", "FREEZE"
             ,"CANNOT", "ALL", "EAST","GIVE-UP","FAMILY","BAD","GREEN","CAN","LEARN","COAT","DRINK","HEAD","FOOTBALL", "LOUSY","BUY", "EXCITED", "PRICE", "ENOUGH", "GRANDMOTHER", "LIE"
             ,"SAUSAGE_OR_HOT-DOG", "THRILL_OR_WHATS-UP", "SHAME", "SAME-OLD", "MESSED-UP", "MATCH", "BAR", "CAR", "HELMET", "ILLEGAL", "MERGE_OR_MAINSTREAM", "CHASE", "WORK-OUT", "WEEKEND"
             ,"DIRTY", "HOW-MANY_OR_MANY", "GONE", "FAR", "HEAD-COLD", "CHAIN_OR_OLYMPICS", "LINE", "GO-AWAY", "COLLECT", "SET-UP", "COUNTRY", "REALLY", "PROTEST", "FLAT-TIRE", "LUNGS"
             ,"PAINT", "INJECT", "EASY", "LIP_OR_MOUTH", "NAB", "FAIL", "FENCE", "TO-FOOL", "GAMBLE", "BANANA", "INTRODUCE", "MOSQUITO", "LEND", "FINALLY", "HALLOWEEN", "EXACT", "HEARING-AID"
             ,"EXPLAIN", "LECTURE", "BICYCLE", "MAGAZINE", "INCREASE", "DISAPPEAR", "MAKE", "LOSE-COMPETITION", "EXPERIENCE", "EXPENSIVE", "GIRL", "ACCEPT", "BUT"]

val_sign_list = ["MEET", "FINISH" ,"ADVISE_OR_INFLUENCE", "COURSE", "DESTROY", "COUGH", "ALONE", "BRIDGE", "CALL-BY-PHONE", "HARD", "IDEA", "APPLE", "HOSPITAL", "ONE-MONTH", "BLACK",
                 "GRASS", "BORROW" ,"RUN-OUT", "BREAD", "MONDAY", "LIBRARY", "ONE", "METAL", "MORNING", "HIT", "MOST", "MEAT","COME-ON", "NOT-MIND", "SMOOTH"]

test_sign_list = ["GENERATION", "HALF", "ENGAGEMENT", "BREAK-DOWN" ,"APPLY", "DATE_OR_DESSERT", "SHAPE_OR_STATUE", "PASS", "INSULT", "HAMBURGER", "OBSCURE", "LIKE", "CRUSH", "LESS-THAN","BAWL-OUT", "BLIND", "PAPER-CHECK_OR_CARD", "GET-UP"
             , "PLACE", "COOPERATE_OR_UNITE", "INSURANCE_OR_INFECTION", "FOLLOW", "MEETING", "GENERAL", "AUTUMN", "COMB", "EXPERIMENT", "LINE-UP", "GAS_OR_GAS-UP", "GRAB-CHANCE", "PERMIT"
             ,"EAT", "TOUGH", "TRASH_OR_BAG", "SPEECH_OR_ORAL", "CHERISH", "STRANGE", "ASSOCIATION", "PULL", "MEMBER", "GHOST", "MACHINE", "AVERAGE", "ACT", "AHEAD","CELEBRATE", "SKIN"
             ,"STRONG", "WHERE", "CONCERN"]

sign_list = ["EXCUSE","INCLUDE_OR_INVOLVE", "SHELF_OR_FLOOR", "ANSWER", "BOSS", "DRESS_OR_CLOTHES", "MARRY", "STAND-UP", "DISAPPOINT", "EXPERT", "CANCEL_OR_CRITICIZE", "FED-UP_OR_FULL"
             ,"GUITAR", "EMPHASIZE", "GOVERNMENT", "LOOK", "AFRAID", "COURT", "MEDICINE", "HELLO", "CONFLICT_OR_INTERSECTION", "LESS", "OF-COURSE", "DISMISS", "DARK", "SILLY", "HOME"
             ,"BLUE", "APPOINTMENT", "DISCONNECT", "A-LOT", "ENTER", "MAD", "COLD", "DECIDE", "ARRIVE", "INFORM", "PROCEED", "MISS_OR_ASSUME", "LETTER_OR_MAIL", "KEEP", "FLY-BY-PLANE"
             ,"DEAF", "HIGH","BEAUTIFUL", "AGAIN", "HAPPEN", "DEPRESS", "EMBARRASS", "DEPOSIT", "DRUNK", "DEVELOP", "OVER_OR_AFTER", "BRAVE_OR_RECOVER", "AVOID_OR_FALL-BEHIND","FULL"
             ,"BLAME", "GOAL", "ART_OR_DESIGN", "ALLOW", "LIVE", "FUTURE","BOY", "NICE_OR_CLEAN", "DRY", "HAVE", "TAKE-UP", "HEAVY", "GROW", "EARTH", "FRIDAY", "DOWN", "BORE", "CENTER"
             ,"CHEAP", "EVERYDAY", "DIVORCE", "FORGET", "AWKWARD", "GRANDFATHER", "CRUEL", "GRADUATE", "BEER", "LEAVE-THERE", "ANY", "CHEMISTRY", "BROWN", "FRIEND", "LEFT", "FREE", "FREEZE"
             ,"CANNOT", "ALL", "EAST","GIVE-UP","FAMILY","BAD","GREEN","CAN","LEARN","COAT","DRINK","HEAD","FOOTBALL", "LOUSY","BUY", "EXCITED", "PRICE", "ENOUGH", "GRANDMOTHER", "LIE"
             ,"SAUSAGE_OR_HOT-DOG", "THRILL_OR_WHATS-UP", "SHAME", "SAME-OLD", "MESSED-UP", "MATCH", "BAR", "CAR", "HELMET", "ILLEGAL", "MERGE_OR_MAINSTREAM", "CHASE", "WORK-OUT", "WEEKEND"
             ,"DIRTY", "HOW-MANY_OR_MANY", "GONE", "FAR", "HEAD-COLD", "CHAIN_OR_OLYMPICS", "LINE", "GO-AWAY", "COLLECT", "SET-UP", "COUNTRY", "REALLY", "PROTEST", "FLAT-TIRE", "LUNGS"
             ,"PAINT", "INJECT", "EASY", "LIP_OR_MOUTH", "NAB", "FAIL", "FENCE", "TO-FOOL", "GAMBLE", "BANANA", "INTRODUCE", "MOSQUITO", "LEND", "FINALLY", "HALLOWEEN", "EXACT", "HEARING-AID"
             ,"EXPLAIN", "LECTURE", "BICYCLE", "MAGAZINE", "INCREASE", "DISAPPEAR", "MAKE", "LOSE-COMPETITION", "EXPERIENCE", "EXPENSIVE", "GIRL", "ACCEPT", "BUT", "MEET", "FINISH"
             ,"ADVISE_OR_INFLUENCE", "COURSE", "DESTROY", "COUGH", "ALONE", "BRIDGE", "CALL-BY-PHONE", "HARD", "IDEA", "APPLE", "HOSPITAL", "ONE-MONTH", "BLACK", "GRASS", "BORROW"
             ,"RUN-OUT", "BREAD", "MONDAY", "LIBRARY", "ONE", "METAL", "MORNING", "HIT", "MOST", "MEAT","COME-ON", "NOT-MIND", "SMOOTH", "GENERATION", "HALF", "ENGAGEMENT", "BREAK-DOWN"
             ,"APPLY", "DATE_OR_DESSERT", "SHAPE_OR_STATUE", "PASS", "INSULT", "HAMBURGER", "OBSCURE", "LIKE", "CRUSH", "LESS-THAN","BAWL-OUT", "BLIND", "PAPER-CHECK_OR_CARD", "GET-UP"
             , "PLACE", "COOPERATE_OR_UNITE", "INSURANCE_OR_INFECTION", "FOLLOW", "MEETING", "GENERAL", "AUTUMN", "COMB", "EXPERIMENT", "LINE-UP", "GAS_OR_GAS-UP", "GRAB-CHANCE", "PERMIT"
             ,"EAT", "TOUGH", "TRASH_OR_BAG", "SPEECH_OR_ORAL", "CHERISH", "STRANGE", "ASSOCIATION", "PULL", "MEMBER", "GHOST", "MACHINE", "AVERAGE", "ACT", "AHEAD","CELEBRATE", "SKIN"
             ,"STRONG", "WHERE", "CONCERN"]


def read_pickle(pickle_file):
    features = []
    with (open(pickle_file, "rb")) as openfile:
        while True:
            try:
                features.append(pickle.load(openfile))
            except EOFError:
                break
    return features[0]


def read_sign_class_desc_feature(location, sign_class):
    sign_desc_loc = location + sign_class.upper() + '.txt'
    with open(sign_desc_loc) as f:
        data = json.load(f)
    feature_vals = np.empty([1, 768])
    data = data["features"][0]["layers"][0]["values"]
    feature_vals[0, :] = data
    return feature_vals


def get_train_val_test_descriptions(textual_features_dir):

    train_sign_class_desc_feature_arr = np.zeros((len(train_sign_list), 768))
    for i, label in enumerate(train_sign_list):
        train_sign_class_desc_feature_arr[i, :] = read_sign_class_desc_feature(textual_features_dir, label)

    val_sign_class_desc_feature_arr = np.zeros((len(val_sign_list), 768))
    for i, label in enumerate(val_sign_list):
        val_sign_class_desc_feature_arr[i, :] = read_sign_class_desc_feature(textual_features_dir, label)

    test_sign_class_desc_feature_arr = np.zeros((len(test_sign_list), 768))
    for i, label in enumerate(test_sign_list):
        test_sign_class_desc_feature_arr[i, :] = read_sign_class_desc_feature(textual_features_dir, label)

    return train_sign_class_desc_feature_arr, val_sign_class_desc_feature_arr, test_sign_class_desc_feature_arr




class ASLLVDDataset(Dataset):
    def __init__(self, vid_pickle_file, hand_pickle_file, split, params):
        vid_label_arr = []
        video_feature_arr = []
        vid_pickle_arr = read_pickle(vid_pickle_file)

        hand_label_arr = []
        hand_feature_arr = []
        hand_pickle_arr = read_pickle(hand_pickle_file)

        text_location = "./features/textual_features_bert_base_uncased/outputs_base/"

        for p in vid_pickle_arr:
            label, feature = p
            vid_label_arr.append(label)
            video_feature_arr.append(feature)
        for p in hand_pickle_arr:
            label, feature = p
            hand_label_arr.append(label)
            hand_feature_arr.append(feature)

        if split == 'train':
            sign_class_desc_feature_arr = np.zeros((len(train_sign_list), 768))
            for i, label in enumerate(train_sign_list):
                sign_class_desc_feature_arr[i, :] = read_sign_class_desc_feature(text_location, label)
        elif split == 'val':
            sign_class_desc_feature_arr = np.zeros((len(val_sign_list), 768))
            for i, label in enumerate(val_sign_list):
                sign_class_desc_feature_arr[i, :] = read_sign_class_desc_feature(text_location, label)
        elif split == 'test':
            sign_class_desc_feature_arr = np.zeros((len(test_sign_list), 768))
            for i, label in enumerate(test_sign_list):
                sign_class_desc_feature_arr[i, :] = read_sign_class_desc_feature(text_location, label)

        self.video_feature_arr = video_feature_arr
        self.vid_label_arr = vid_label_arr
        self.hand_feature_arr = hand_feature_arr
        self.hand_label_arr = hand_label_arr
        self.sign_class_dec_feature_arr = sign_class_desc_feature_arr
        self.split = split
        self.params = params

        print "----------"
        print "video len = ", len(self.video_feature_arr)
        print "label len = ", len(self.vid_label_arr)
        print "hand video len = ", len(self.hand_feature_arr)
        print "hand label len = ", len(self.hand_label_arr)
        print "sign class desc len = ", len(self.sign_class_dec_feature_arr)
        print "unique sign desc len = ", len(set(self.vid_label_arr))
        print "hand vid label arr equal? = ", np.array_equal(self.vid_label_arr, self.hand_label_arr)
        print "----------"

    def __len__(self):
        return len(self.vid_label_arr)

    def __getitem__(self, idx):
        vid_feature = self.video_feature_arr[idx]
        hand_feature = self.hand_feature_arr[idx]
        vid_label = self.vid_label_arr[idx]
        if self.split == 'train':
            label_index = train_sign_list.index(vid_label)
        elif self.split == 'val':
            label_index = val_sign_list.index(vid_label)
        elif self.split == 'test':
            label_index = test_sign_list.index(vid_label)

        vid_feature = torch.from_numpy(vid_feature).float()
        hand_feature = torch.from_numpy(hand_feature).float()

        vid_feature = vid_feature.view(-1, 1024)
        hand_feature = hand_feature.view(-1, 1024)


        sample = {'vid_feature': vid_feature, 'hand_feature':hand_feature, 'vid_label': vid_label, "vid_label_index": label_index}
        return sample


def fetch_dataloader(types, params, data_dir_vid, data_dir_hand):
    train_vid_val = 'train_vid_5c.pickle'
    train_hand_val = 'train_hand_5c.pickle'
    val_vid_val = 'val_vid_5c.pickle'
    val_hand_val = 'val_hand_5c.pickle'
    test_vid_val = 'test_vid_5c.pickle'
    test_hand_val = 'test_hand_5c.pickle'


    dataloaders = {}
    for split in ['train', 'val', 'test']:
        if split in types:
            if split == 'train':
                vid_path = os.path.join(data_dir_vid, train_vid_val)
                hand_path = os.path.join(data_dir_hand, train_hand_val)
                print "train = ", vid_path, " - ", hand_path
                dl = DataLoader(ASLLVDDataset(vid_path, hand_path, "train", params), batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
            elif split == 'val':
                vid_path = os.path.join(data_dir_vid, val_vid_val)
                hand_path = os.path.join(data_dir_hand, val_hand_val)
                print "val = ", vid_path, " - ", hand_path
                dl = DataLoader(ASLLVDDataset(vid_path, hand_path, "val", params), batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
            elif split == 'test':
                vid_path = os.path.join(data_dir_vid, test_vid_val)
                hand_path = os.path.join(data_dir_hand, test_hand_val)
                print "test = ", vid_path, " - ", hand_path
                dl = DataLoader(ASLLVDDataset(vid_path, hand_path, "test", params), batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
            dataloaders[split] = dl

    return dataloaders
