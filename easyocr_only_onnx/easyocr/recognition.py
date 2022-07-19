from PIL import Image
import numpy as np
from .utils import CTCLabelConverter
from scipy.special import softmax
import onnxruntime
import time
import math

def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

class ListDataset():

    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, 'L')
    
class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        imgN = np.expand_dims(np.asarray(img).astype(np.float), axis=0) / 255.0
        imgN = (imgN - 0.5) / 0.5

        c, h, w = imgN.shape
        Pad_imgN = np.zeros(self.max_size).astype(np.float)
        Pad_imgN[:, :, :w] = imgN  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_imgN[:, :, w:] = np.broadcast_to(np.expand_dims(imgN[:, :, w - 1], axis=2),(c, h, self.max_size[2] - w))
        
        return Pad_imgN

class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, adjust_contrast = 0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images = batch
        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            #### augmentation here - change contrast
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L"))
                image = adjust_contrast_grey(image, target = self.adjust_contrast)
                image = Image.fromarray(image, 'L')

            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = np.concatenate([np.expand_dims(t, axis=0) for t in resized_images], axis=0)
        
        return image_tensors

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def recognizer_predict(model, converter, test_loader, batch_max_length,\
                       ignore_idx, char_group_idx, decoder = 'greedy', beamWidth= 5, device = 'cpu', img_list = []):
    
    result = []
    for image_tensors in test_loader[0]:

     
        # Usar modelo ONNX
        start_time = time.time()
        ort_session = onnxruntime.InferenceSession("onnx_models/recognition_model.onnx")
        # print("--- %s seconds ---" % (time.time() - start_time))
        
        start_time = time.time()
        ort_inputs = {ort_session.get_inputs()[0].name: image_tensors[np.newaxis,np.newaxis,...].astype(np.single)}
        # print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        # print("--- %s seconds ---" % (time.time() - start_time))
        
        ######## filter ignore_char, rebalance
        preds_prob = softmax(ort_outs[0], axis=2)
        preds_prob[:,:,ignore_idx] = 0.
        pred_norm = preds_prob.sum(axis=2)
        preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)

        if decoder == 'greedy':
            # Select max probabilty (greedy decoding) then decode index to character
            preds_index = preds_prob.argmax(2)
            preds_str = converter.decode_greedy(preds_index[0], [preds_prob.shape[1]])
        elif decoder == 'beamsearch':
            preds_str = converter.decode_beamsearch(preds_prob, beamWidth=beamWidth)
        elif decoder == 'wordbeamsearch':
            preds_str = converter.decode_wordbeamsearch(preds_prob, beamWidth=beamWidth)

        values = preds_prob.max(axis=2)
        indices = preds_prob.argmax(axis=2)
        preds_max_prob = []
        for v,i in zip(values, indices):
            max_probs = v[i!=0]
            if len(max_probs)>0:
                preds_max_prob.append(max_probs)
            else:
                preds_max_prob.append(np.array([0]))

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            confidence_score = custom_mean(pred_max_prob)
            result.append([pred, confidence_score])

    return result

def get_recognizer(recog_network, network_params, character,\
                   separator_list, dict_list, model_path,\
                   device = 'cpu', quantize = True):

    converter = CTCLabelConverter(character, separator_list, dict_list)

    return None, converter

def get_text(character, imgH, imgW, recognizer, converter, image_list,\
             ignore_char = '',decoder = 'greedy', beamWidth =5, batch_size=1, contrast_ths=0.1,\
             adjust_contrast=0.5, filter_ths = 0.003, workers = 1, device = 'cpu'):
    batch_max_length = int(imgW/10)
    
    char_group_idx = {}
    ignore_idx = []
    for char in ignore_char:
        try: ignore_idx.append(character.index(char)+1)
        except: pass

    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    test_data = ListDataset(img_list)
    AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    test_loader = AlignCollate_normal(test_data)
    # predict first round
    result1 = recognizer_predict(None, converter, test_loader, batch_max_length,\
                                 ignore_idx, char_group_idx, decoder, beamWidth, device = device, img_list=img_list)

    # predict second round
    low_confident_idx = [i for i,item in enumerate(result1) if (item[1] < contrast_ths)]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        test_data = ListDataset(img_list2)
        AlignCollate_contrast = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True, adjust_contrast=adjust_contrast)
        test_loader = AlignCollate_contrast(test_data)
        result2 = recognizer_predict(None, converter, test_loader, batch_max_length,\
                                     ignore_idx, char_group_idx, decoder, beamWidth, device = device, img_list=img_list2)

    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1]>pred2[1]:
                result.append( (box, pred1[0], pred1[1]) )
            else:
                result.append( (box, pred2[0], pred2[1]) )
        else:
            result.append( (box, pred1[0], pred1[1]) )

    return result
