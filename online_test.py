# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing

import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
import json
import os
from multiprocessing import Pool
import shutil
import glob

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default="8")
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default="/MEVA_TSM_LABEL/test_videofolder.txt")
parser.add_argument('--actev',action="store_true", default=True)

parser.add_argument('--softmax', default=True, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--topk', type=int, default=37)
parser.add_argument('--class_num', type=int, default=37)
parser.add_argument('--out_path', type=str, default="/results")
parser.add_argument('--if_prepared',default = True,action="store_true")
parser.add_argument('--prop_path',type = str,default = "/props")
parser.add_argument('--data_path',type = str,default = "/imgs")
parser.add_argument('--inner_data_path',type = str,default = "/inner_imgs")
parser.add_argument('--output_filename',type = str,default = "output.json")



args = parser.parse_args()


def get_prop_move(vid,prop_id,args):
    json_file = json.load(open(os.path.join(args.prop_path,vid,"annotation",os.path.splitext(vid)[0],"actv_id_type.json"),"r"))
    if prop_id not in list(json_file.keys()):
        print(prop_id)
        print(vid)
        raise RuntimeError("prop not in json_file")
    start_frame = json_file[prop_id]["start_frame"]
    return int(start_frame)

def relocate_vid_api(opt):
    vid = opt[0]
    args = opt[1]
    num = opt[2]
    relocate_vids(vid,args,num)
    return


def relocate_vids(vid,args,num):
    print(vid)
    print(num)
    vid_path = os.path.join(args.data_path,vid)
    frames = glob.glob(vid_path+"/*.jpg")
    # frames = os.listdir(vid_path)
    new_vid_name = vid.split("_")[0]+".avi"
    prop_id = vid.split("_")[1]

    if len(frames)<10:
        print("frames less than 10")
        return
    if len(frames)<90:
        frames.sort()
        mov = get_prop_move(new_vid_name,prop_id,args)
        start = str(mov)
        end = str(mov+len(frames))
        split_name = start+"_"+end+"_"+prop_id
        if not os.path.exists(os.path.join(args.inner_data_path,new_vid_name,split_name)):
            os.makedirs(os.path.join(args.inner_data_path,new_vid_name,split_name))
        # f.write(os.path.join(new_vid_name,split_name)+" "+str(len(frames))+" 0"+"\n")
        for i in range(len(frames)):
            frame = frames[i]
            new_frame = "img_"+str(i+1).zfill(5)+".jpg"
            # new_frame = "img_"+frame.split("_")[1]
            src = os.path.join(args.data_path,vid,frame)
            det = os.path.join(args.inner_data_path,new_vid_name,split_name,new_frame)
            os.system("cp "+src+" "+det)
            # shutil.copy(os.path.join(args.data_path,vid,frame),os.path.join(args.inner_data_path,new_vid_name,split_name,new_frame))

    else:
        frames.sort()
        # print(frames)
        epocs = len(frames) // 90
        mov = get_prop_move(new_vid_name,prop_id,args)
        for epoc in range(epocs):
            if len(frames)-epoc*90<30:
                break
            start = str(epoc*90+mov)
            end = str(epoc*90+mov+min(90,len(frames)-epoc*90))
            split_name = start+"_"+end+"_"+prop_id
            if not os.path.exists(os.path.join(args.inner_data_path,new_vid_name,split_name)):
                os.makedirs(os.path.join(args.inner_data_path,new_vid_name,split_name))
            # f.write(os.path.join(new_vid_name,split_name)+" "+str(min(90,len(frames)-epoc*90))+" 0"+"\n")
            for i in range(epoc*90,min(len(frames),(epoc+1)*90)):
                frame = frames[i]
                new_frame = "img_"+str(i-epoc*90+1).zfill(5)+".jpg"
                # new_frame = "img_"+frame.split("_")[1]
                src = os.path.join(args.data_path,vid,frame)
                det = os.path.join(args.inner_data_path,new_vid_name,split_name,new_frame)
                os.system("cp "+src+" "+det)
                # shutil.copy(os.path.join(args.data_path,vid,frame),os.path.join(args.inner_data_path,new_vid_name,split_name,new_frame))
    return

def prepare_data(args):
    n_jobs=128
    vids = os.listdir(args.data_path)
    print(len(vids))
    args_list = []
    num_list = []
    for i in range(len(vids)):
        args_list.append(args)
        num_list.append(i)
    opts_list = list(zip(vids,args_list,num_list))
    pool = Pool(n_jobs)
    pool.map(relocate_vid_api, opts_list)
    pool.close()

def prepare_testlist(args):
    f = open(args.test_list,"w")
    file_path = ""
    if args.if_prepared:
        file_path = args.data_path
    else:
        file_path = args.inner_data_path
    vids = os.listdir(file_path)
    for vid in vids:
        splits = os.listdir(os.path.join(file_path,vid))
        for split in splits:
            f.write(os.path.join(vid,split)+" "+str(len(os.listdir(os.path.join(file_path,vid,split))))+" 0"+"\n")
    return



if not args.if_prepared:
    prepare_data(args)
prepare_testlist(args)

event_dict = ["heu_negative","person_closes_facility_door","person_closes_vehicle_door","Closing_Trunk","person_enters_through_structure","person_enters_vehicle","person_exits_through_structure","person_exits_vehicle","person_loads_vehicle","Open_Trunk","person_opens_facility_door","person_opens_vehicle_door","Transport_HeavyCarry","person_unloads_vehicle","vehicle_turning_left","vehicle_turning_right","vehicle_u_turn","Riding","Talking","specialized_talking_phone","specialized_texting_phone","person_sitting_down","person_standing_up","person_reading_document","object_transfer","person_picks_up_object","person_sets_down_object","hand_interaction","person_person_embrace","person_purchasing","person_laptop_interaction","vehicle_stopping","vehicle_starting","vehicle_reversing","vehicle_picks_up_person","vehicle_drops_off_person","abandon_package"]

vehicle_events_list = ["vehicle_turning_left","vehicle_turning_right","vehicle_u_turn","vehicle_stopping","vehicle_starting","vehicle_reversing"]
person_events_list = ["person_closes_facility_door","person_enters_through_structure","person_exits_through_structure","person_opens_facility_door",
"Talking","specialized_talking_phone","specialized_texting_phone","person_picks_up_object","person_sets_down_object","hand_interaction","person_person_embrace","person_purchasing","person_laptop_interaction","abandon_package","Riding","person_sitting_down","person_standing_up","person_reading_document","object_transfer"]
person_vehicle_interact_list = ["person_closes_vehicle_door","Closing_Trunk","person_enters_vehicle","person_exits_vehicle","person_loads_vehicle","Open_Trunk","person_opens_vehicle_door","Transport_HeavyCarry","Unloading","vehicle_picks_up_person","vehicle_drops_off_person","person_unloads_vehicle"]

if args.topk > args.class_num:
    print(args.topk)
    print(args.class_num)
    raise RuntimeError("class number missmatch")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def getoutput(vid_names,video_pred_topall,video_prob_topall,event_dict,prop_path):
    new_dict = {}
    new_dict["filesProcessed"] = []
    new_dict["activities"] = []
    act_list = []
    props = os.listdir(prop_path)
    for prop in props:
        if prop not in new_dict["filesProcessed"]:
            new_dict["filesProcessed"].append(prop)

    for vid_name in vid_names:
        prefix_name = vid_name.split("/")[0]
        if prefix_name not in new_dict["filesProcessed"]:
            new_dict["filesProcessed"].append(prefix_name)
    for name, pred_all, prob_all in zip(vid_names, video_pred_topall,video_prob_topall):
        prefix_name = name.split("/")[0]
        start_frame = name.split("/")[1].split("_")[0]
        end_frame = name.split("/")[1].split("_")[1]
        prop_id = name.split("/")[1].split("_")[2]
        for i in range(len(pred_all)):
            pred = pred_all[i]
            event = event_dict[int(pred)]
            if event=="heu_negative":
                continue
            prob = prob_all[i]
            if event not in act_list and event!="heu_negative":
                act_list.append(event)
            act_dict = {}
            act_dict["activity"] = event
            act_dict["activityID"] = int(pred)
            act_dict["presenceConf"] = float(prob)
            start = start_frame
            end = end_frame
                # raise RuntimeError("stop")
            act_dict["localization"] = {prefix_name:{start:1,end:0}}
            act_dict["proposal_id"] = prop_id
            new_dict["activities"].append(act_dict)
    file_dict = get_file_index(new_dict["filesProcessed"])
    eve_dict = get_activity_index(act_list)
    return new_dict,file_dict,eve_dict
            

def get_obj_types(event):
    if event in vehicle_events_list:
        return ["Vehicle"]
    elif event in person_events_list:
        return ["Person"]
    elif event in person_vehicle_interact_list:
        return ["Vehicle","Person"]
    else:
        print (event)
        raise RuntimeError("unseen events")

def get_activity_index(activities):
    new_dict = {}
    for act in activities:
        new_dict[act] = {"objectTypes":get_obj_types(act)}
    return new_dict

def get_file_index(filesProcessed):
    new_dict = {}
    for f in filesProcessed:
        new_dict[f]={"framerate": 30.0, "selected": {"0": 1, "9000": 0}}
    return new_dict




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)


data_iter_list = []
net_list = []
modality_list = []

total_num = None
for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
    if 'RGB' in this_weights:
        modality = 'RGB'
    else:
        modality = 'Flow'
    this_arch = this_weights.split('TSM_')[1].split('_')[2]
    modality_list.append(modality)
    num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                            modality)
    if args.if_prepared:
        root_path = args.data_path
    else:
        root_path = args.inner_data_path
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
    net = TSN(num_class, this_test_segments if is_shift else 1, modality,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights,
              )

    if 'tpool' in this_weights:
        from ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

    checkpoint = torch.load(this_weights)
    checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)

    net.load_state_dict(base_dict)

    input_size = net.scale_size if args.full_res else net.input_size
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(root_path, test_file if test_file is not None else val_list, num_segments=this_test_segments,
                       new_length=1 if modality == "RGB" else 5,
                       modality=modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
    )

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda())
    net.eval()

    data_gen = enumerate(data_loader)

    if total_num is None:
        total_num = len(data_loader.dataset)
    else:
        assert total_num == len(data_loader.dataset)

    data_iter_list.append(data_gen)
    net_list.append(net)


output = []


def eval_video(video_data, net, this_test_segments, modality):
    net.eval()
    with torch.no_grad():
        i, data, label = video_data
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.twice_sample:
            num_crop *= 2

        if modality == 'RGB':
            length = 3
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        data_in = data.view(-1, length, data.size(2), data.size(3))
        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        if net.module.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return i, rst, label


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

# top1 = AverageMeter()
# top5 = AverageMeter()

for i, data_label_pairs in enumerate(zip(*data_iter_list)):
    with torch.no_grad():
        if i >= max_num:
            break
        this_rst_list = []
        this_label = None
        for n_seg, (_, (data, label)), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
            rst = eval_video((i, data, label), net, n_seg, modality)
            this_rst_list.append(rst[1])
            this_label = label
        assert len(this_rst_list) == len(coeff_list)
        for i_coeff in range(len(this_rst_list)):
            this_rst_list[i_coeff] *= coeff_list[i_coeff]
        ensembled_predict = sum(this_rst_list) / len(this_rst_list)

        for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
            output.append([p[None, ...], g])
        cnt_time = time.time() - proc_start_time
        # prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))
        # top1.update(prec1.item(), this_label.numel())
        # top5.update(prec5.item(), this_label.numel())
        if i % 20 == 0:
            # print(len(output))
            # video_pred_topall = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in output]
            # video_prob_topall = [np.sort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in output]
            # print(video_pred_topall)
            # print(video_prob_topall)
            # print(output[0])
            # raise RuntimeError("snip test")
            print('video {} done, total {}/{}, average {:.3f} sec/video, '.format(i * args.batch_size, i * args.batch_size, total_num,float(cnt_time) / (i+1) / args.batch_size))

video_pred = [np.argmax(x[0]) for x in output]
# video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

#update to contain probability
video_pred_topall = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in output]
video_prob_topall = [np.sort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in output]



video_labels = [x[1] for x in output]



if args.actev:
    with open(test_file_list[0]) as f:
        vid_names = f.readlines()
    vid_names = [n.split(' ')[0] for n in vid_names]
    assert len(vid_names) == len(video_pred)
    output_dict,file_dict,eve_dict = getoutput(vid_names,video_pred_topall,video_prob_topall,event_dict,args.prop_path)
    json_str = json.dumps(output_dict,indent=4)
    with open(os.path.join(args.out_path,args.output_filename), 'w') as save_json:
        save_json.write(json_str)
    json_str = json.dumps(file_dict,indent=4)
    with open(os.path.join(args.out_path,"file-index.json"), 'w') as save_json:
        save_json.write(json_str)
    json_str = json.dumps(eve_dict,indent=4)
    with open(os.path.join(args.out_path,"activity-index.json"), 'w') as save_json:
        save_json.write(json_str)
    



# if args.csv_file is not None:
#     print('=> Writing result to csv file: {}'.format(args.csv_file))
#     with open(test_file_list[0].replace('test_videofolder.txt', 'category.txt')) as f:
#         categories = f.readlines()
#     categories = [f.strip() for f in categories]
#     with open(test_file_list[0]) as f:
#         vid_names = f.readlines()
#     vid_names = [n.split(' ')[0] for n in vid_names]
#     assert len(vid_names) == len(video_pred)
#     if args.dataset != 'MEVA':  # only output top1
#         with open(args.csv_file, 'w') as f:
#             for n, pred in zip(vid_names, video_pred):
#                 f.write('{};{}\n'.format(n, categories[pred]))
#     else:
#         with open(args.csv_file, 'w') as f:
#             for n, pred5 in zip(vid_names, video_pred_topall):
#                 fill = [n]
#                 for p in list(pred5):
#                     fill.append(p)
#                 f.write('{};{};{};{};{};{}\n'.format(*fill))


# cf = confusion_matrix(video_labels, video_pred).astype(float)

# np.save('cm.npy', cf)
# cls_cnt = cf.sum(axis=1)
# cls_hit = np.diag(cf)

# cls_acc = cls_hit / cls_cnt
# print(cls_acc)
# upper = np.mean(np.max(cf, axis=1) / cls_cnt)
# print('upper bound: {}'.format(upper))

# print('-----Evaluation is finished------')
# print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
# print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))


