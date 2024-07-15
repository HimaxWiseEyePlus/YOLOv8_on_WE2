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
#include "Person_Detect_Model.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"

#include <inttypes.h>
/* Helper macro to convert RGB888 to RGB565 format. */
#define RGB888_TO_RGB565(R8,G8,B8)  ((((R8>>3) & 0x1F) << 11) |     \
                                     (((G8>>2) & 0x3F) << 5)  |     \
                                     ((B8>>3) & 0x1F))

constexpr uint16_t COLOR_BLACK  = 0;
constexpr uint16_t COLOR_GREEN  = RGB888_TO_RGB565(  0, 255,  0); // 2016;
constexpr uint16_t COLOR_YELLOW = RGB888_TO_RGB565(255, 255,  0); // 65504;

using ImgClassClassifier = arm::app::Classifier;

namespace arm {
namespace app {

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
    * @brief           Helper function to convert the current RGB image to Gray image 
    *                  and load the Gray image into the input tensor.
    * @param[in]       imIdx         Image index (from the pool of images available
    *                                to the application).
    * @param[out]      inputTensor   Pointer to the input tensor to be populated.
    * @return          true if tensor is loaded, false otherwise.
    **/
    static bool LoadRGBTOGREYImageIntoTensor(uint32_t imIdx, TfLiteTensor* inputTensor);
    /**
    * @brief           Helper function to convert the current RGB image to Gray image.
    * @param[in]       srcPtr        Pointer to the input RGB image.
    * @param[in]       dstImgSz      Output image size
    * @param[out]      dstPtr        Pointer to the Output Gray image.
    **/
    void RgbToGrayscale(const uint8_t* srcPtr, uint8_t* dstPtr, const size_t dstImgSz);
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

    /* Image inference classification handler. */
    bool ClassifyImageHandler(ApplicationContext& ctx, uint32_t imgIndex, bool runAll)
    {
        auto& platform = ctx.Get<hal_platform&>("platform");
        auto& profiler = ctx.Get<Profiler&>("profiler");

        constexpr uint32_t dataPsnImgDownscaleFactor = 1;
        constexpr uint32_t dataPsnImgStartX = 10;
        constexpr uint32_t dataPsnImgStartY = 35;

        constexpr uint32_t dataPsnTxtInfStartX = 150;
        constexpr uint32_t dataPsnTxtInfStartY = 40;

        platform.data_psn->clear(COLOR_BLACK);

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
        TfLiteTensor* output=nullptr;
        output = model.GetOutputTensor(0);
        TfLiteTensor* inputTensor = model.GetInputTensor(0);
        if (!inputTensor->dims) {
            printf_err("Invalid input tensor dims\n");
            return false;
        } else if (inputTensor->dims->size < 3) {
            printf_err("Input tensor dimension should be >= 3\n");
            return false;
        }

        TfLiteIntArray* inputShape = model.GetInputShape(0);

        const uint32_t nCols = inputShape->data[arm::app::Person_Detect_Model::ms_inputColsIdx];
        const uint32_t nRows = inputShape->data[arm::app::Person_Detect_Model::ms_inputRowsIdx];
        const uint32_t nChannels = inputShape->data[arm::app::Person_Detect_Model::ms_inputChannelsIdx];
         info("tensor input:col:%d, row:%d, ch:%d\n", nCols,nRows,nChannels);
        std::vector<ClassificationResult> results;

        do {
            /* Strings for presentation/logging. */
            std::string str_inf{"Running inference... "};

            /* Copy over the data. */
            LoadRGBTOGREYImageIntoTensor(ctx.Get<uint32_t>("imgIndex"), inputTensor);
            /* Display this image on the LCD. */
            platform.data_psn->present_data_image(
                (uint8_t*) inputTensor->data.uint8,
                nCols, nRows, nChannels,
                dataPsnImgStartX, dataPsnImgStartY, dataPsnImgDownscaleFactor);

            /* If the data is signed. */
            if (model.IsDataSigned()) {
                ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
            }
            printf("\n"); 
            /* Display message on the LCD - inference running. */
            platform.data_psn->present_data_text(str_inf.c_str(), str_inf.size(),
                                    dataPsnTxtInfStartX, dataPsnTxtInfStartY, 0);

            /* Run inference over this image. */
            info("Running inference on image %" PRIu32 " => %s\n", ctx.Get<uint32_t>("imgIndex"),
                get_filename(ctx.Get<uint32_t>("imgIndex")));

            if (!RunInference(model, profiler)) {
                return false;
            }

            /* Erase. */
            str_inf = std::string(str_inf.size(), ' ');
            platform.data_psn->present_data_text(str_inf.c_str(), str_inf.size(),
                                    dataPsnTxtInfStartX, dataPsnTxtInfStartY, 0);

            info("output->type: %d\n",output->type);
            info("output->bytes: %d\n",output->bytes);
            int8_t person_score = output->data.int8[1];
            int8_t no_person_score = output->data.int8[0];
            info("person_score: %d no_person_score: %d\n",person_score,no_person_score);
           
            uint8_t grey_img[96*96]={0};
            const uint8_t* image = get_img_array(ctx.Get<uint32_t>("imgIndex"));
            RgbToGrayscale(image,grey_img,96*96);
            platform.data_psn->present_data_image(
                (uint8_t*) grey_img,
                nCols, nRows, nChannels,
                dataPsnImgStartX, dataPsnImgStartY, dataPsnImgDownscaleFactor);
            

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
    void RgbToGrayscale(const uint8_t* srcPtr, uint8_t* dstPtr, const size_t dstImgSz)
    {
        const float R = 0.299;
        const float G = 0.587;
        const float B = 0.114;
        for (size_t i = 0; i < dstImgSz; ++i, srcPtr += 3) {
            uint32_t  int_gray = R * (*srcPtr) +
                                 G * (*(srcPtr + 1)) +
                                 B * (*(srcPtr + 2));
            *dstPtr++ = int_gray <= std::numeric_limits<uint8_t>::max() ?
                        int_gray : std::numeric_limits<uint8_t>::max();
        }
    }
    static bool LoadRGBTOGREYImageIntoTensor(uint32_t imIdx, TfLiteTensor* inputTensor)
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
        RgbToGrayscale(imgSrc, inputTensor->data.uint8, copySz);
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

    

    static void ConvertImgToInt8(void* data, const size_t kMaxImageSize)
    {
        auto* tmp_req_data = (uint8_t*) data;
        auto* tmp_signed_req_data = (int8_t*) data;

        for (size_t i = 0; i < kMaxImageSize; i++) {
            tmp_signed_req_data[i] = (int8_t) (
                (int32_t) (tmp_req_data[i]) - 128);
        }
    }

} /* namespace app */
} /* namespace arm */
