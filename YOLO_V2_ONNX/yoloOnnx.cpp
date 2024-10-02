/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.yolo_onnx";


const std::string VOC_class_names[] = {
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
};

// Structure pour stocker les informations d'un objet détecté
struct BoundingBox {
    cv::Rect bbox;
    float confidence;
    int classId;
};

//! \brief  The YoloOnnx class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class YoloOnnx
{
public:
    YoloOnnx(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    float sigmoid(float x);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool YoloOnnx::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto timingCache = SampleUniquePtr<nvinfer1::ITimingCache>();
       
    auto constructed = constructNetwork(builder, network, config, parser, timingCache);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    if (timingCache != nullptr && !mParams.timingCacheFile.empty())
    {
        samplesCommon::updateTimingCacheFile(
            sample::gLogger.getTRTLogger(), mParams.timingCacheFile, timingCache.get(), *builder);
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 4);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx Yolo Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx Yolo network
//!
//! \param builder Pointer to the engine builder
//!
bool YoloOnnx::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.bf16)
    {
        config->setFlag(BuilderFlag::kBF16);
    }
    if (mParams.int8) {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 0.0F, 255.0F);
    }
    if (mParams.timingCacheFile.size())
    {
        timingCache = samplesCommon::buildTimingCacheFromFile(
            sample::gLogger.getTRTLogger(), *config, mParams.timingCacheFile, sample::gLogError);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    // Add optimization profile
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    const char* inputTensorName = network->getInput(0)->getName();
    profile->setDimensions(inputTensorName, OptProfileSelector::kMIN, Dims4(1, 3, 416, 416)); // minimal size
    profile->setDimensions(inputTensorName, OptProfileSelector::kOPT, Dims4(1, 3, 416, 416)); // optimal size
    profile->setDimensions(inputTensorName, OptProfileSelector::kMAX, Dims4(1, 3, 416, 416)); // maximal size
    config->addOptimizationProfile(profile);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool YoloOnnx::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool YoloOnnx::processInput(const samplesCommon::BufferManager& buffers) {
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::string imgFilePath = locateFile(mParams.imgFileName, mParams.dataDirs).c_str();
    cv::Mat image = cv::imread(imgFilePath);

    if (image.empty()) {
        sample::gLogError << "Error: Image not loaded.Check the file path : " << imgFilePath << std::endl;
        return false;
    }

    if (image.channels() != 3) {
        sample::gLogError << "Error: Image must have 3 channels (RGB)." << std::endl;
        return false;
    }

    cv::imshow("Input image", image);
    cv::waitKey(0);

    cv::resize(image, image, cv::Size(inputW, inputH));

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // Copy data in the buffer
    for (int i = 0; i < inputH * inputW * 3; ++i)
    {
        hostDataBuffer[i] = static_cast<float>(image.data[i]) / 255.0f;; // normalize the data
    }

    return true;
}

//!
//! \brief Extracts outputs of the model and creates the bounding box by evaluating the confidence
//!

bool YoloOnnx::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    std::string imgFilePath = locateFile(mParams.imgFileName, mParams.dataDirs).c_str();
    cv::Mat inputImage = cv::imread(imgFilePath);

    const int gridSize = 13;
    const int numClasses = 20;
    const int numAnchors = 5;
    const int outputSize = gridSize * gridSize * (numAnchors * (5 + numClasses)); // 125 * 13 * 13

    // Anchor boxes
    std::vector<cv::Vec2f> anchors = { {1.08, 1.19}, {3.42, 4.41}, {6.63, 11.38}, {9.42, 5.11}, {16.62, 10.52} };

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    // Iterate over the grid
    for (int cy = 0; cy < gridSize; ++cy)
    {
        for (int cx = 0; cx < gridSize; ++cx)
        {
            for (int b = 0; b < numAnchors; ++b)
            {
                int index = (cy * gridSize + cx) * (numAnchors * (5 + numClasses)) + b * (5 + numClasses);

                // Extract bounding box parameters
                float tx = output[index];
                float ty = output[index + 1];
                float tw = output[index + 2];
                float th = output[index + 3];
                float confidence = output[index + 4];

                // Calculate bounding box position relative to original image size
                float x = (cx + sigmoid(tx)) / gridSize * inputImage.cols;
                float y = (cy + sigmoid(ty)) / gridSize * inputImage.rows;
                float w = anchors[b][0] * exp(tw) * inputImage.cols / gridSize;
                float h = anchors[b][1] * exp(th) * inputImage.rows / gridSize;

                // Find the best class
                float bestClassProb = 0;
                int bestClass = 0;
                for (int c = 0; c < numClasses; ++c)
                {
                    float classProb = sigmoid(output[index + 5 + c]);
                    if (classProb > bestClassProb)
                    {
                        bestClassProb = classProb;
                        bestClass = c;
                    }
                }

                // Filter out low-confidence boxes
                if (confidence * bestClassProb > 0.5)
                {
                    cv::Rect box(x - w / 2, y - h / 2, w, h);
                    boxes.push_back(box);
                    confidences.push_back(confidence * bestClassProb);
                    classIds.push_back(bestClass);
                }
            }
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

    // Draw boxes on the image
    for (int idx : indices)
    {
        cv::rectangle(inputImage, boxes[idx], cv::Scalar(0, 255, 0), 2);
        cv::putText(inputImage, VOC_class_names[classIds[idx]], boxes[idx].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Display the image
    cv::imshow("image output", inputImage);
    cv::waitKey(0);

    return true;
}


float YoloOnnx::sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}


//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/images/");
        params.dataDirs.push_back("model/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "tinyyolov2-8.onnx";
    params.imgFileName = "dog_and_bicycle.jpeg";
    params.inputTensorNames.push_back("image");
    params.outputTensorNames.push_back("grid");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.bf16 = args.runInBf16;
    params.timingCacheFile = args.timingCacheFile;

    return params;
}


//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << "[-t or --timingCacheFile=<path to timing cache file]" << std::endl;
    std::cout << "--help             Display help information" << std::endl;
    std::cout << "--datadir          Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N     Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8             Run in Int8 mode." << std::endl;
    std::cout << "--fp16             Run in FP16 mode." << std::endl;
    std::cout << "--bf16             Run in BF16 mode." << std::endl;
    std::cout << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
              << "created." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    YoloOnnx sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx YOLOV2" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
