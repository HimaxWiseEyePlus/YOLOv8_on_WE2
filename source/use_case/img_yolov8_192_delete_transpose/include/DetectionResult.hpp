/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
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
#ifndef DETECTION_RESULT_HPP
#define DETECTION_RESULT_HPP


namespace arm {
namespace app {
namespace object_detection {

    /**
     * @brief   Class representing a single detection result.
     */
    class DetectionResult {
    public:
        /**
         * @brief       Constructor
         * @param[in]   normalisedVal   Result normalized value
         * @param[in]   x0              Top corner x starting point
         * @param[in]   y0              Top corner y starting point
         * @param[in]   w               Detection result width
         * @param[in]   h               Detection result height
         **/
        DetectionResult(int class_d,double normalisedVal,int x0,int y0, int w,int h) :
                m_class(class_d),
                m_normalisedVal(normalisedVal),
                m_x0(x0),
                m_y0(y0),
                m_w(w),
                m_h(h)
            {
            }

        DetectionResult() = default;
        ~DetectionResult() = default;
        int     m_class{0};
        double  m_normalisedVal{0.0};
        int     m_x0{0};
        int     m_y0{0};
        int     m_w{0};
        int     m_h{0};
    };

} /* namespace object_detection */
} /* namespace app */
} /* namespace arm */

#endif /* DETECTION_RESULT_HPP */
