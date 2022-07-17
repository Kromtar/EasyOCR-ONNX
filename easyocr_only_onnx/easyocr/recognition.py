import numpy as np
from .utils import CTCLabelConverter
from scipy.special import softmax
import onnxruntime
import time

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def recognizer_predict(model, converter, test_loader, batch_max_length,\
                       ignore_idx, char_group_idx, decoder = 'greedy', beamWidth= 5, device = 'cpu', img_list = []):
    
    result = []

    for image_tensors in img_list:

    
        # Usar modelo ONNX
        start_time = time.time()
        ort_session = onnxruntime.InferenceSession("onnx_models/recognition_model.onnx")
        print("--- %s seconds ---" % (time.time() - start_time))
        
        start_time = time.time()
        ort_inputs = {ort_session.get_inputs()[0].name: image_tensors[np.newaxis,np.newaxis,...].astype(np.single)}
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        ######## filter ignore_char, rebalance
        preds_prob = softmax(ort_outs[0], axis=2)
        preds_prob[:,:,ignore_idx] = 0.
        pred_norm = preds_prob.sum(axis=2)
        preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)

        '''
        batch_size = 1
        preds = torch.from_numpy(ort_outs[0])
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds_prob = torch.from_numpy(preds_prob).float().to(device)
        if decoder == 'greedy':
            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds_prob.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)
        elif decoder == 'beamsearch':
            k = preds_prob.cpu().detach().numpy()
            preds_str = converter.decode_beamsearch(k, beamWidth=beamWidth)
        elif decoder == 'wordbeamsearch':
            k = preds_prob.cpu().detach().numpy()
            preds_str = converter.decode_wordbeamsearch(k, beamWidth=beamWidth)
        preds_prob = preds_prob.cpu().detach().numpy()
        '''

        #preds_str = converter.decode_beamsearch(preds_prob, beamWidth=beamWidth)
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

    # predict first round
    result1 = recognizer_predict(None, converter, None,batch_max_length,\
                                 ignore_idx, char_group_idx, decoder, beamWidth, device = device, img_list=img_list)

    # predict second round
    low_confident_idx = [i for i,item in enumerate(result1) if (item[1] < contrast_ths)]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]

        result2 = recognizer_predict(None, converter, None, batch_max_length,\
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
