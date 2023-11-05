import openslide 
from PIL.ImageDraw import ImageDraw
import predict_module
import os
import argparse
import tqdm

def iou(box1, box2):
    # box=[left, top, right, bottom]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = area1 + area2 - inter_area

    return max(inter_area / area1, inter_area / area2, inter_area / union_area)

#非极大抑制函数
def nms(boxes_list, threshold):
    '''
    boxes_list: 一个列表，包含所有的矩形框，每个矩形框由4个坐标值和一个置信度值组成\n
    threshold: 一个阈值，表示IoU的阈值
    '''
    new_boxes_list = []
    boxes_list.sort(key=lambda x: x[-1], reverse=True)
    while boxes_list:
        max_box = boxes_list.pop(0)
        new_boxes_list.append(max_box)
        temp_boxes_list = []
        for box in boxes_list:
            iou_value = iou(max_box, box)
            if iou_value < threshold:
                temp_boxes_list.append(box)
        boxes_list = temp_boxes_list
    return new_boxes_list

def change_box(box,x,y,level=1):
    m=4**(level-1)
    return [(box[0]+x*300)*m,(box[1]+y*300)*m,(box[2]+x*300)*m,(box[3]+y*300)*m,box[4]]

def predict(args):
    model_path = args.model_path
    save_folder = args.save_folder
    slide_name=os.path.basename(args.slide_path)
    slide_name,slide_type = os.path.splitext(slide_name)

    print('reading slide...')
    oslide = openslide.open_slide(args.slide_path)
    w1,h1 = oslide.level_dimensions[1]
    w2,h2 = oslide.level_dimensions[2]
    pre_result = oslide.read_region((0,0),1,(w1,h1))

    print('creating model...')
    myModule = predict_module.FPNrcnn(model_path=model_path,
                                        num_classes=2,batch_size=64,num_workers=0)
    box_list = []
    cur=tqdm.tqdm(total=int(w1//300)*int(h1//300)+int(w2//300)*int(h2//300))

    for x in range(w1 // 300):
        for y in range(h1 // 300):
            if True:
                patch = oslide.read_region(((x * 300)*4, (y * 300)*4),1,(600,600))
                results = myModule.predict(patch)
                for result in results:
                    box = change_box(result,x,y)
                    box_list.append(box)

            cur.update(1)
    for x in range (w2 // 300):
        for y in range (h2 // 300):
            if True:
                patch = oslide.read_region(((x * 300)*16, (y * 300)*16),2,(600,600))
                results = myModule.predict(patch)
                for result in results:
                    box = change_box(result,x,y,2)
                    box_list.append(box)
            cur.update(1)

    box_list = nms(box_list,0.5)
    draw = ImageDraw(pre_result)
    for box in box_list:
        draw.rectangle(box[:4],outline="#00FF00",width=10)
    mask_file_name = 'result_'+ slide_name +'.png'
    pre_result.save(save_folder+mask_file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', type=str)
    parser.add_argument('--model_path', type=str, default='./models/FPNrcnn.pth')
    parser.add_argument('--save_folder', type=str, default='./save_folder/')
    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    predict(args)