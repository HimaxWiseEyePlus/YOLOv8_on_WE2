/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "UseCaseHandler.hpp"

#include "Classifier.hpp"
#include "InputFiles.hpp"
#include "Yolov8nOd.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"

#include <inttypes.h>
/////////////////////////
#include "DetectionResult.hpp"
#include "ImageUtils.hpp"
#include <forward_list>
#include <algorithm>
#include <math.h>
#include <stdint.h>
#define MAX_TRACKED_ALGO_RES  10
#define COLOR_DEPTH	1 // 8bit per pixel FU
typedef enum
{
	MONO_FRAME=0,
	RAWBAYER_FRAME,
	YUV_FRAME
}enum_frameFormat;


typedef struct
{
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
}struct__box;

typedef struct
{
	struct__box bbox;
    uint32_t time_of_existence;
    uint32_t is_reliable;
}struct_MotionTarget;



typedef struct
{
	struct__box upper_body_bbox;
    uint32_t upper_body_scale;
    uint32_t upper_body_score;
    uint32_t upper_body_num_frames_since_last_redetection_attempt;
    struct__box head_bbox;
    uint32_t head_scale;
    uint32_t head_score;
    uint32_t head_num_frames_since_last_redetection_attempt;
    uint32_t octave;
    uint32_t time_of_existence;
    uint32_t isVerified;
}struct_Human;

typedef struct
{
    int num_hot_pixels ;
    struct_MotionTarget Emt[MAX_TRACKED_ALGO_RES]; ; //ecv::motion::Target* *tracked_moving_targets;
    int frame_count ;
    short num_tracked_moving_targets;
    short num_tracked_human_targets ;
    bool humanPresence ;
    struct_Human ht[MAX_TRACKED_ALGO_RES];  //TrackedHumanTarget* *tracked_human_targets;
    int num_reliable_moving_targets;
    int verifiedHumansExist;
}struct_algoResult;

typedef struct boxabs {
    float left, right, top, bot;
} boxabs;


typedef struct branch {
    int resolution;
    int num_box;
    float *anchor;
    int8_t *tf_output;
    float scale;
    int zero_point;
    size_t size;
    float scale_x_y;
} branch;

typedef struct network {
    int input_w;
    int input_h;
    int num_classes;
    int num_branch;
    branch *branchs;
    int topN;
} network;


typedef struct box {
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    float *prob;
    float objectness;
} detection;

struct_algoResult algoresult;



typedef struct detection_yolov8{
    box bbox;
    float confidence;
    float index;

} detection_yolov8;

std::string coco_classes[] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};
int coco_ids[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
                      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                      57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
                      86, 87, 88, 89, 90};

#define RGB888_TO_RGB565(R8,G8,B8)  ((((R8>>3) & 0x1F) << 11) |     \
                                     (((G8>>2) & 0x3F) << 5)  |     \
                                     ((B8>>3) & 0x1F))

uint16_t COLOR_RED  = RGB888_TO_RGB565(  255, 0,  0);
//////////////////////



using ImgClassClassifier = arm::app::Classifier;

namespace arm {
namespace app {


/////////////////////////////////////
    static int sort_class;

void free_dets(std::forward_list<detection> &dets){
    std::forward_list<detection>::iterator it;
    for ( it = dets.begin(); it != dets.end(); ++it ){
        free(it->prob);
    }
}

float sigmoid(float x)
{
    return 1.f/(1.f + exp(-x));
} 

bool det_objectness_comparator(detection &pa, detection &pb)
{
    return pa.objectness < pb.objectness;
}

void insert_topN_det(std::forward_list<detection> &dets, detection det){
    std::forward_list<detection>::iterator it;
    std::forward_list<detection>::iterator last_it;
    for ( it = dets.begin(); it != dets.end(); ++it ){
        if(it->objectness > det.objectness)
            break;
        last_it = it;
    }
    if(it != dets.begin()){
        dets.emplace_after(last_it, det);
        free(dets.begin()->prob);
        dets.pop_front();
    }
    else{
        free(det.prob);
    }
}

std::forward_list<detection> get_network_boxes(network *net, int image_w, int image_h, float thresh, int *num)
{
    std::forward_list<detection> dets;
    int i;
    int num_classes = net->num_classes;
    *num = 0;

    for (i = 0; i < net->num_branch; ++i) {
        int height  = net->branchs[i].resolution;
        int width = net->branchs[i].resolution;
        int channel  = net->branchs[i].num_box*(5+num_classes);

        for (int h = 0; h < net->branchs[i].resolution; h++) {
            for (int w = 0; w < net->branchs[i].resolution; w++) {
                for (int anc = 0; anc < net->branchs[i].num_box; anc++) {
                    
                    // objectness score
                    int bbox_obj_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 4;
                    float objectness = sigmoid(((float)net->branchs[i].tf_output[bbox_obj_offset] - net->branchs[i].zero_point) * net->branchs[i].scale);

                    if(objectness > thresh){
                        detection det;
                        det.prob = (float*)calloc(num_classes, sizeof(float));
                        det.objectness = objectness;
                        //get bbox prediction data for each anchor, each feature point
                        int bbox_x_offset = bbox_obj_offset -4;
                        int bbox_y_offset = bbox_x_offset + 1;
                        int bbox_w_offset = bbox_x_offset + 2;
                        int bbox_h_offset = bbox_x_offset + 3;
                        int bbox_scores_offset = bbox_x_offset + 5;
                        //int bbox_scores_step = 1;
                        det.bbox.x = ((float)net->branchs[i].tf_output[bbox_x_offset] - net->branchs[i].zero_point) * net->branchs[i].scale;
                        det.bbox.y = ((float)net->branchs[i].tf_output[bbox_y_offset] - net->branchs[i].zero_point) * net->branchs[i].scale;
                        det.bbox.w = ((float)net->branchs[i].tf_output[bbox_w_offset] - net->branchs[i].zero_point) * net->branchs[i].scale;
                        det.bbox.h = ((float)net->branchs[i].tf_output[bbox_h_offset] - net->branchs[i].zero_point) * net->branchs[i].scale;

                        float bbox_x, bbox_y;

                        // Eliminate grid sensitivity trick involved in YOLOv4
                        bbox_x = sigmoid(det.bbox.x); //* net->branchs[i].scale_x_y - (net->branchs[i].scale_x_y - 1) / 2;
                        bbox_y = sigmoid(det.bbox.y); //* net->branchs[i].scale_x_y - (net->branchs[i].scale_x_y - 1) / 2;
                        det.bbox.x = (bbox_x + w) / width;
                        det.bbox.y = (bbox_y + h) / height;

                        det.bbox.w = exp(det.bbox.w) * net->branchs[i].anchor[anc*2] / net->input_w;
                        det.bbox.h = exp(det.bbox.h) * net->branchs[i].anchor[anc*2+1] / net->input_h;
                        
                        for (int s = 0; s < num_classes; s++) {
                            det.prob[s] = sigmoid(((float)net->branchs[i].tf_output[bbox_scores_offset + s] - net->branchs[i].zero_point) * net->branchs[i].scale)*objectness;
                            det.prob[s] = (det.prob[s] > thresh) ? det.prob[s] : 0;
                        }

                        //correct_yolo_boxes 
                        det.bbox.x *= image_w;
                        det.bbox.w *= image_w;
                        det.bbox.y *= image_h;
                        det.bbox.h *= image_h;

                        if (*num < net->topN || net->topN <=0){
                            dets.emplace_front(det);
                            *num += 1;
                        }
                        else if(*num ==  net->topN){
                            dets.sort(det_objectness_comparator);
                            insert_topN_det(dets,det);
                            *num += 1;
                        }else{
                            insert_topN_det(dets,det);
                        }
                    }
                }
            }
        }
    }
    if(*num > net->topN)
        *num -=1;
    return dets;
}

// init part

branch create_brach(int resolution, int num_box, float *anchor, int8_t *tf_output, size_t size, float scale, int zero_point){
    branch b;
    b.resolution = resolution;
    b.num_box = num_box;
    b.anchor = anchor;
    b.tf_output = tf_output;
    b.size = size;
    b.scale = scale;
    b.zero_point = zero_point;
    return b;
}

network creat_network(int input_w, int input_h, int num_classes, int num_branch, branch* branchs, int topN){
    network net;
    net.input_w = input_w;
    net.input_h = input_h;
    net.num_classes = num_classes;
    net.num_branch = num_branch;
    net.branchs = branchs;
    net.topN = topN;
    return net;
}

// NMS part

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    float I = box_intersection(a, b);
    float U = box_union(a, b);
    if (I == 0 || U == 0) {
        return 0;
    }
    return I / U;
}

bool det_comparator(detection &pa, detection &pb)
{
    return pa.prob[sort_class] > pb.prob[sort_class];
}

void do_nms_sort(std::forward_list<detection> &dets, int classes, float thresh)
{
    int k;
    
    for (k = 0; k < classes; ++k) {
        sort_class = k;
        dets.sort(det_comparator);
        
        for (std::forward_list<detection>::iterator it=dets.begin(); it != dets.end(); ++it){
            if (it->prob[k] == 0) continue;
            for (std::forward_list<detection>::iterator itc=std::next(it, 1); itc != dets.end(); ++itc){
                if (itc->prob[k] == 0) continue;
                if (box_iou(it->bbox, itc->bbox) > thresh) {
                    itc->prob[k] = 0;
                }
            }
        }
    }
}


boxabs box_c(box a, box b) {
    boxabs ba;//
    ba.top = 0;
    ba.bot = 0;
    ba.left = 0;
    ba.right = 0;
    ba.top = fmin(a.y - a.h / 2, b.y - b.h / 2);
    ba.bot = fmax(a.y + a.h / 2, b.y + b.h / 2);
    ba.left = fmin(a.x - a.w / 2, b.x - b.w / 2);
    ba.right = fmax(a.x + a.w / 2, b.x + b.w / 2);
    return ba;
}


float box_diou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, 0.6);
    float diou_term = u;

    return iou - diou_term;
}


void diounms_sort(std::forward_list<detection> &dets, int classes, float thresh)
{
    int k;
    
    for (k = 0; k < classes; ++k) {
        sort_class = k;
        dets.sort(det_comparator);
        
        for (std::forward_list<detection>::iterator it=dets.begin(); it != dets.end(); ++it){
            if (it->prob[k] == 0) continue;
            for (std::forward_list<detection>::iterator itc=std::next(it, 1); itc != dets.end(); ++itc){
                if (itc->prob[k] == 0) continue;
                if (box_diou(it->bbox, itc->bbox) > thresh) {
                    itc->prob[k] = 0;
                }
            }
        }
    }
}


bool yolov8_det_comparator(detection_yolov8 &pa, detection_yolov8 &pb)
{
    return pa.confidence > pb.confidence;
}

/*
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
static float yolov8_box_iou(box a, box b)
{
    int xx_a = MAX(a.x , b.x);
    int yy_a = MAX(a.y , b.y);
    int xx_b = MIN((a.x + a.w - 1), (b.x + b.w - 1));
    int yy_b = MIN((a.y + a.h -1), (b.y + b.h -1));

    int insection_w, insection_h;
    insection_w = MAX(0 , (xx_b - xx_a + 1));
    insection_h = MAX(0 , (yy_b - yy_a + 1));

    float insection_area, union_area, iou;
    insection_area = float(insection_w) * insection_h;
    union_area = float( a.w * a.h + b.w * b.h - insection_area);
    iou = insection_area / union_area;
    return iou;

}         
*/


// void  yolov8_NMSBoxes(std::vector<box> &boxes,std::vector<float> &confidences,float modelScoreThreshold,float modelNMSThreshold,std::vector<int>& nms_result)
// {
//     detection_yolov8 yolov8_bbox;
//     std::vector<detection_yolov8> yolov8_bboxes{};
//     for(int i = 0; i < boxes.size(); i++)
//     {
//         yolov8_bbox.bbox = boxes[i];
//         yolov8_bbox.confidence = confidences[i];
//         yolov8_bbox.index = i;
//         yolov8_bboxes.push_back(yolov8_bbox);
//     }
//     sort(yolov8_bboxes.begin(), yolov8_bboxes.end(), yolov8_det_comparator);
//     int updated_size = yolov8_bboxes.size();
//     for(int k = 0; k < updated_size; k++)
//     {
//         if(yolov8_bboxes[k].confidence < modelScoreThreshold)
//         {
//             continue;
//         }
        
//         nms_result.push_back(yolov8_bboxes[k].index);
//         for(int j = k + 1; j < updated_size; j++)
//         {
//             float iou = box_iou(yolov8_bboxes[k].bbox, yolov8_bboxes[j].bbox);
//             // float iou = yolov8_box_iou(yolov8_bboxes[k].bbox, yolov8_bboxes[j].bbox);
//             // float iou = box_diou(yolov8_bboxes[k].bbox, yolov8_bboxes[j].bbox);
//             if(iou > modelNMSThreshold)
//             {
//                 yolov8_bboxes.erase(yolov8_bboxes.begin() + j);
//                 updated_size = yolov8_bboxes.size();
//                 j = j -1;
//             }
//         }

//     }
// }


void  yolov8_NMSBoxes(std::vector<box> &boxes,std::vector<float> &confidences,float modelScoreThreshold,float modelNMSThreshold,std::vector<int>& nms_result)
{
    detection_yolov8 yolov8_bbox;
    std::vector<detection_yolov8> yolov8_bboxes{};
    for(int i = 0; i < boxes.size(); i++)
    {
        yolov8_bbox.bbox = boxes[i];
        yolov8_bbox.confidence = confidences[i];
        yolov8_bbox.index = i;
        yolov8_bboxes.push_back(yolov8_bbox);
    }
    sort(yolov8_bboxes.begin(), yolov8_bboxes.end(), yolov8_det_comparator);
    int updated_size = yolov8_bboxes.size();
    for(int k = 0; k < updated_size; k++)
    {
        if(yolov8_bboxes[k].confidence < modelScoreThreshold)
        {
            continue;
        }
        
        nms_result.push_back(yolov8_bboxes[k].index);
        for(int j = k + 1; j < updated_size; j++)
        {
            float iou = box_iou(yolov8_bboxes[k].bbox, yolov8_bboxes[j].bbox);
            // float iou = box_diou(yolov8_bboxes[k].bbox, yolov8_bboxes[j].bbox);
            if(iou > modelNMSThreshold)
            {
                yolov8_bboxes.erase(yolov8_bboxes.begin() + j);
                updated_size = yolov8_bboxes.size();
                j = j -1;
            }
        }

    }
}

///////////////////////////////////////////////////////////////

    /**
    * @brief           Helper function to load the current image into the input
    *                  tensor.
    * @param[in]       imIdx         Image index (from the pool of images available
    *                                to the application).
    * @param[out]      inputTensor   Pointer to the input tensor to be populated.
    * @return          true if tensor is loaded, false otherwise.
    **/
    static bool LoadImageIntoTensor(uint32_t imIdx, TfLiteTensor* inputTensor);

    /**
     * @brief           Helper function to increment current image index.
     * @param[in,out]   ctx   Pointer to the application context object.
     **/
    static void IncrementAppCtxImageIdx(ApplicationContext& ctx);

    /**
     * @brief           Helper function to set the image index.
     * @param[in,out]   ctx   Pointer to the application context object.
     * @param[in]       idx   Value to be set.
     * @return          true if index is set, false otherwise.
     **/
    static bool SetAppCtxImageIdx(ApplicationContext& ctx, uint32_t idx);
    /**
     * @brief           Helper function to convert a UINT8 image to INT8 format.
     * @param[in,out]   data            Pointer to the data start.
     * @param[in]       kMaxImageSize   Total number of pixels in the image.
     **/
    static void ConvertImgToInt8(void* data, size_t kMaxImageSize);
    /**
     * @brief           Draw boxes directly on the LCD for all detected objects.
     * @param[in]       platform           Reference to the hal platform object.
     * @param[in]       results            Vector of detection results to be displayed.
     * @param[in]       imageStartX        X coordinate where the image starts on the LCD.
     * @param[in]       imageStartY        Y coordinate where the image starts on the LCD.
     * @param[in]       imgDownscaleFactor How much image has been downscaled on LCD.
     **/
    static void DrawDetectionBoxes(hal_platform& platform,const std::vector<object_detection::DetectionResult>& results,
                                   uint32_t imgStartX,
                                   uint32_t imgStartY,
                                   uint32_t imgDownscaleFactor);
    /**
     * @brief           Presents inference results along using the data presentation
     *                  object.
     * @param[in]       platform           Reference to the hal platform object.
     * @param[in]       results            Vector of detection results to be displayed.
     * @return          true if successful, false otherwise.
     **/
    static bool PresentInferenceResult(hal_platform& platform,const std::vector<object_detection::DetectionResult>& results);
    /* Image inference classification handler. */
    bool ClassifyImageHandler(ApplicationContext& ctx, uint32_t imgIndex, bool runAll)
    {
        auto& platform = ctx.Get<hal_platform&>("platform");
        auto& profiler = ctx.Get<Profiler&>("profiler");

        constexpr uint32_t dataPsnImgDownscaleFactor = 1;//2;//
        constexpr uint32_t dataPsnImgStartX = 10;
        constexpr uint32_t dataPsnImgStartY = 35;
        //constexpr uint32_t dataPsnImgStartX = 10;
        // constexpr uint32_t dataPsnImgStartY = 0;

        constexpr uint32_t dataPsnTxtInfStartX = 150;
        constexpr uint32_t dataPsnTxtInfStartY = 40;

        auto& model = ctx.Get<Model&>("model");

        /* If the request has a valid size, set the image index. */
        if (imgIndex < NUMBER_OF_FILES) {
            if (!SetAppCtxImageIdx(ctx, imgIndex)) {
                return false;
            }
        }
        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        auto curImIdx = ctx.Get<uint32_t>("imgIndex");
        const size_t numOutputs = model.GetNumOutputs();
        info("Num of outputs: %zu\n",numOutputs);
        TfLiteTensor* output;
        TfLiteTensor* output_2;
        output = model.GetOutputTensor(0);
        output_2 = model.GetOutputTensor(1);
        
        TfLiteTensor* inputTensor = model.GetInputTensor(0);

        if (!inputTensor->dims) {
            printf_err("Invalid input tensor dims\n");
            return false;
        } else if (inputTensor->dims->size < 3) {
            printf_err("Input tensor dimension should be >= 3\n");
            return false;
        }

        TfLiteIntArray* inputShape = model.GetInputShape(0);

        const uint32_t nCols = inputShape->data[arm::app::Yolov8nOd::ms_inputColsIdx];
        const uint32_t nRows = inputShape->data[arm::app::Yolov8nOd::ms_inputRowsIdx];
        const uint32_t nChannels = inputShape->data[arm::app::Yolov8nOd::ms_inputChannelsIdx];
        info("tensor input:col:%d, row:%d, ch:%d\n", nCols,nRows,nChannels);
        std::vector<object_detection::DetectionResult> results;

        do {
            platform.data_psn->clear(COLOR_BLACK);

            /* Copy over the data. */
            LoadImageIntoTensor(ctx.Get<uint32_t>("imgIndex"), inputTensor);

            /* Display this image on the LCD. */
            const uint8_t* image = get_img_array(ctx.Get<uint32_t>("imgIndex"));
            platform.data_psn->present_data_image(
                (uint8_t*) image,
                nCols, nRows, nChannels,
                dataPsnImgStartX, dataPsnImgStartY, dataPsnImgDownscaleFactor);
            
            /* If the data is signed. */
            if (model.IsDataSigned()) {
                ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
            }
            printf("\n"); 
            /* Run inference over this image. */
            info("Running YOLO v8 192x192x3 model inference on image %" PRIu32 " => %s\n", ctx.Get<uint32_t>("imgIndex"),
                get_filename(ctx.Get<uint32_t>("imgIndex")));

            if (!RunInference(model, profiler)) {
                return false;
            }

            ////////////////////////////
            //////////////////////
            int input_w=192;
            int input_h=192;
            // init postprocessing 	
            int num_classes = output_2->dims->data[2];

            
            // end init
            ///////////////////////
            // start postprocessing
            int nboxes=0;
            float modelScoreThreshold = .25;
            float modelNMSThreshold = .45;
            // float modelScoreThreshold = .5;
            // float modelNMSThreshold = .45;

            // float modelScoreThreshold = .4;
            // float modelNMSThreshold = .45;
            int image_width = input_w;
            int image_height =  input_h;
            
            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<box> boxes;
            // (output->data.int8 - (static_cast<TfLiteAffineQuantization*>(output->quantization.params)->zero_point->data[0])) *  (static_cast<TfLiteAffineQuantization*>(output->quantization.params))->scale->data[0]
            printf("output->dims->size: %d\r\n",output->dims->size);
            float output_scale = ((TfLiteAffineQuantization*)(output->quantization.params))->scale->data[0];
            int output_zeropoint = ((TfLiteAffineQuantization*)(output->quantization.params))->zero_point->data[0];
            int output_size = output->bytes;

            float output_2_scale = ((TfLiteAffineQuantization*)(output_2->quantization.params))->scale->data[0];
	        int output_2_zeropoint = ((TfLiteAffineQuantization*)(output_2->quantization.params))->zero_point->data[0];

            printf("output_scale: %f\r\n",output_scale);
            printf("output_zeropoint: %d\r\n",output_zeropoint);
            printf("output_size: %d\r\n",output_size);
            printf("output->dims->data[0]: %d\r\n",output->dims->data[0]);//1
            printf("output->dims->data[1]: %d\r\n",output->dims->data[1]);//84
            printf("output->dims->data[2]: %d\r\n",output->dims->data[2]);//756

            float outputs_data[output->dims->data[2]][output->dims->data[1]];//do transpose about output

            /***
             * dequantize the output result for box
             * 
             * 
             ******/
            for(int dims_cnt_2 = 0; dims_cnt_2 < output->dims->data[2]; dims_cnt_2++)//// output->dims->data[2] is 756
            {
                float outputs_bbox_data[4];
                float maxScore = (-1);// the first four indexes are bbox information
                uint16_t maxClassIndex = 0;
                for(int dims_cnt_1 = 0; dims_cnt_1 < output->dims->data[1]; dims_cnt_1++)// output->dims->data[1] is 4 
                {
                    int value =  output->data.int8[ dims_cnt_2 + dims_cnt_1 * output->dims->data[2]];
                    
                    float deq_value = ((float) value-(float)output_zeropoint) * output_scale ;

                    /***
                     * fix big score
                     * ****/
                    if(dims_cnt_1%2)//==1
                    {
                        deq_value *= (float)input_h;
                    }
                    else
                    {
                        deq_value *= (float)input_w;
                    }
                    outputs_bbox_data[dims_cnt_1] = deq_value;
                }

                for(int output_2_dims_cnt_1 = 0; output_2_dims_cnt_1 < output_2->dims->data[2]; output_2_dims_cnt_1++)//output_2->dims->data[2] is 80
                {
                    int value_2 =  output_2->data.int8[ output_2_dims_cnt_1 + dims_cnt_2 * output_2->dims->data[2]];
                    
                    float deq_value_2 = ((float) value_2-(float)output_2_zeropoint) * output_2_scale ;
                    /***
                     * find maximum Score and correspond Class idx
                     * **/
                    if(maxScore < deq_value_2)
                    {
                        maxScore = deq_value_2;
                        maxClassIndex = output_2_dims_cnt_1;
                    }
                }
                if (maxScore >= modelScoreThreshold)
                {
                    box bbox;
                    
                    bbox.x = (outputs_bbox_data[0] - (0.5 * outputs_bbox_data[2]));
                    bbox.y = (outputs_bbox_data[1] - (0.5 * outputs_bbox_data[3]));
                    bbox.w =(outputs_bbox_data[2]);
                    bbox.h = (outputs_bbox_data[3]);
                    boxes.push_back(bbox);
                    class_ids.push_back(maxClassIndex);
                    confidences.push_back(maxScore);
                    
                }
            }
            printf("boxes.size(): %d\r\n",boxes.size());

            /**
             * do nms
             * **/

            std::vector<int> nms_result;
            yolov8_NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
            std::vector<object_detection::DetectionResult> results;
            for (int i = 0; i < nms_result.size(); i++)
            {
                int idx = nms_result[i];

                object_detection::DetectionResult temp_result;
                temp_result.m_normalisedVal = confidences[idx];
                temp_result.m_x0 = boxes[idx].x;
                temp_result.m_y0 = boxes[idx].y;
                temp_result.m_w = boxes[idx].w;
                temp_result.m_h = boxes[idx].h;
                // results.push_back(temp_result);
                results.push_back(object_detection::DetectionResult(class_ids[idx],confidences[idx],boxes[idx].x,boxes[idx].y,boxes[idx].w,boxes[idx].h));
                printf("detect object[%d]: %s confidences: %f\r\n",i, coco_classes[class_ids[idx]].c_str(),confidences[idx]);
                // float iou = box_iou(boxes[0], boxes[idx]);
                //  printf("iou: %f modelNMSThreshold: %f\r\n",iou, modelNMSThreshold);
            }
            DrawDetectionBoxes(platform,results, dataPsnImgStartX, dataPsnImgStartY, dataPsnImgDownscaleFactor);
            printf("nms_result.size():%d\r\n",nms_result.size());
            // //////////////////////////////////////////
#if VERIFY_TEST_OUTPUT
            arm::app::DumpTensor(outputTensor);
#endif /* VERIFY_TEST_OUTPUT */
            
            if (!PresentInferenceResult(platform,results)) {
                return false;
            }

            profiler.PrintProfilingResult();

            IncrementAppCtxImageIdx(ctx);

        } while (runAll && ctx.Get<uint32_t>("imgIndex") != curImIdx);

        return true;
    }
    static bool LoadImageIntoTensor(uint32_t imIdx, TfLiteTensor* inputTensor)
    {
        const size_t copySz = inputTensor->bytes < IMAGE_DATA_SIZE ?
                              inputTensor->bytes : IMAGE_DATA_SIZE;
        const uint8_t* imgSrc = get_img_array(imIdx);
        if (nullptr == imgSrc) {
            printf_err("Failed to get image index %" PRIu32 " (max: %u)\n", imIdx,
                       NUMBER_OF_FILES - 1);
            return false;
        }
        info("inputTensor->dims: %d\n",inputTensor->dims->size );
        info("inputTensor->bytes: %d\n",inputTensor->bytes);
        info("IMAGE_DATA_SIZE: %d\n",copySz);
        memcpy(inputTensor->data.data, imgSrc, copySz);
        debug("Image %" PRIu32 " loaded\n", imIdx);
        return true;
    }

    static void IncrementAppCtxImageIdx(ApplicationContext& ctx)
    {
        auto curImIdx = ctx.Get<uint32_t>("imgIndex");

        if (curImIdx + 1 >= NUMBER_OF_FILES) {
            ctx.Set<uint32_t>("imgIndex", 0);
            return;
        }
        ++curImIdx;
        ctx.Set<uint32_t>("imgIndex", curImIdx);
    }

    static bool SetAppCtxImageIdx(ApplicationContext& ctx, uint32_t idx)
    {
        if (idx >= NUMBER_OF_FILES) {
            printf_err("Invalid idx %" PRIu32 " (expected less than %u)\n",
                       idx, NUMBER_OF_FILES);
            return false;
        }
        ctx.Set<uint32_t>("imgIndex", idx);
        return true;
    }

    static bool PresentInferenceResult(hal_platform& platform,const std::vector<object_detection::DetectionResult>& results)
    {
        platform.data_psn->set_text_color(COLOR_GREEN);
        
        /* If profiling is enabled, and the time is valid. */
        info("Final results:\n");
        info("Total number of inferences: 1\n");

        for (uint32_t i = 0; i < results.size(); ++i) {
            info("%" PRIu32 ") (%f) -> %s {x=%d,y=%d,w=%d,h=%d}\n", i,
                results[i].m_normalisedVal, "Detection box:",
                results[i].m_x0, results[i].m_y0, results[i].m_w, results[i].m_h );
        }

        return true;
    }

    static void ConvertImgToInt8(void* data, const size_t kMaxImageSize)
    {
        auto* tmp_req_data = (uint8_t*) data;
        auto* tmp_signed_req_data = (int8_t*) data;

        for (size_t i = 0; i < kMaxImageSize; i++) {
            tmp_signed_req_data[i] = (int8_t) (
                (int32_t) (tmp_req_data[i]) - 128);
        }
    }
    static void DrawDetectionBoxes(hal_platform& platform,const std::vector<object_detection::DetectionResult>& results,
                                   uint32_t imgStartX,
                                   uint32_t imgStartY,
                                   uint32_t imgDownscaleFactor)
    {
        uint32_t lineThickness = 1;

        for (const auto& result: results) {
            //std::string str_inf = std::to_string(result.m_class);
            std::string str_inf = coco_classes[result.m_class];
            platform.data_psn->present_data_text(str_inf.c_str(),str_inf.size(),imgStartX + (result.m_x0+1)/imgDownscaleFactor,
                    imgStartY + (result.m_y0+1)/imgDownscaleFactor,false);
            /* Top line. */
            platform.data_psn->present_box(imgStartX + result.m_x0/imgDownscaleFactor,
                    imgStartY + result.m_y0/imgDownscaleFactor,
                    result.m_w/imgDownscaleFactor, lineThickness, COLOR_RED);
            /* Bot line. */
            platform.data_psn->present_box(imgStartX + result.m_x0/imgDownscaleFactor,
                    imgStartY + (result.m_y0 + result.m_h)/imgDownscaleFactor - lineThickness,
                    result.m_w/imgDownscaleFactor, lineThickness, COLOR_RED);

            /* Left line. */
            platform.data_psn->present_box(imgStartX + result.m_x0/imgDownscaleFactor,
                    imgStartY + result.m_y0/imgDownscaleFactor,
                    lineThickness, result.m_h/imgDownscaleFactor, COLOR_RED);
            /* Right line. */
            platform.data_psn->present_box(imgStartX + (result.m_x0 + result.m_w)/imgDownscaleFactor - lineThickness,
                    imgStartY + result.m_y0/imgDownscaleFactor,
                    lineThickness, result.m_h/imgDownscaleFactor, COLOR_RED);
        }
    }
} /* namespace app */
} /* namespace arm */
