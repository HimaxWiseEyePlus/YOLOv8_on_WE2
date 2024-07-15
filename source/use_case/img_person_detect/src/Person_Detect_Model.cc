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
#include "Person_Detect_Model.hpp"

#include "hal.h"

const tflite::MicroOpResolver& arm::app::Person_Detect_Model::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::Person_Detect_Model::EnlistOperations()
{
    this->m_opResolver.AddAveragePool2D();
    this->m_opResolver.AddConv2D(tflite::Register_CONV_2D_INT8REF());
    this->m_opResolver.AddDepthwiseConv2D();
    this->m_opResolver.AddReshape();
    this->m_opResolver.AddSoftmax();

#if defined(ARM_NPU)
    if (kTfLiteOk == this->m_opResolver.AddEthosU()) {
        info("Added %s support to op resolver\n",
            tflite::GetString_ETHOSU());
    } else {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
#endif 
/* ARM_NPU */
    return true;
}
#include <stdio.h>
extern uint8_t* GetModelPointer();
const uint8_t* arm::app::Person_Detect_Model::ModelPointer()
{
    printf("GetModelPointer(): %x\n",GetModelPointer());
    return GetModelPointer();
}

extern size_t GetModelLen();
size_t arm::app::Person_Detect_Model::ModelSize()
{
    return GetModelLen();
}

