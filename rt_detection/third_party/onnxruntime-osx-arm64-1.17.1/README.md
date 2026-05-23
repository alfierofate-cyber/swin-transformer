<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

**ONNX Runtime 是一个跨平台的推理与训练机器学习加速器**。

**ONNX Runtime 推理** 可以实现更快的用户体验和更低的成本，支持来自 PyTorch、TensorFlow/Keras 等深度学习框架以及 scikit-learn、LightGBM、XGBoost 等经典机器学习库的模型。ONNX Runtime 兼容不同的硬件、驱动和操作系统，并通过硬件加速器以及图优化和图变换来提供最佳性能。[了解更多 &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-inferencing)

**ONNX Runtime 训练** 可以通过在现有 PyTorch 训练脚本中添加一行代码，在多节点 NVIDIA GPU 上加速 Transformer 模型的训练时间。[了解更多 &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-training)

## 入门与资源

* **通用信息**: [onnxruntime.ai](https://onnxruntime.ai)

* **使用文档与教程**: [onnxruntime.ai/docs](https://onnxruntime.ai/docs)

* **YouTube 视频教程**: [youtube.com/@ONNXRuntime](https://www.youtube.com/@ONNXRuntime)

* [**即将发布的版本路线图**](https://github.com/microsoft/onnxruntime/wiki/Upcoming-Release-Roadmap)

* **配套示例仓库**:
  - ONNX Runtime 推理示例: [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples)
  - ONNX Runtime 训练示例: [microsoft/onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples)

## 内置流水线状态

|系统|推理|训练|
|---|---|---|
|Windows|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20TensorRT%20CI%20Pipeline?label=Windows+GPU+TensorRT)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=47)||
|Linux|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20Minimal%20Build%20E2E%20CI%20Pipeline?label=Linux+CPU+Minimal+Build)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=64)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20TensorRT%20CI%20Pipeline?label=Linux+GPU+TensorRT)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=45)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20OpenVINO%20CI%20Pipeline?label=Linux+OpenVINO)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=55)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-ci-pipeline?label=Linux+CPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=86)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-gpu-ci-pipeline?label=Linux+GPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=84)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining/orttraining-ortmodule-distributed?label=Training+Distributed)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=148)|
|Mac|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)||
|Android|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Android%20CI%20Pipeline?label=Android)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=53)||
|iOS|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/iOS%20CI%20Pipeline?label=iOS)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=134)||
|Web|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/ONNX%20Runtime%20Web%20CI%20Pipeline?label=Web)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=161)||
|其他|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/onnxruntime-binary-size-checks-ci-pipeline?repoName=microsoft%2Fonnxruntime&label=Binary+Size+Check)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=187&repoName=microsoft%2Fonnxruntime)||

## 第三方流水线状态

|系统|推理|训练|
|---|---|---|
|Linux|[![Build Status](https://github.com/Ascend/onnxruntime/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/Ascend/onnxruntime/actions/workflows/build-and-test.yaml)||

## 数据/遥测

本项目的 Windows 发行版可能会收集使用数据并发送给 Microsoft，以帮助我们改进产品和服务。详见[隐私声明](docs/Privacy.md)。

## 贡献与反馈

我们欢迎社区贡献！请参阅[贡献指南](CONTRIBUTING.md)。

如需功能请求或 Bug 报告，请提交 [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues)。

如需一般性讨论或提问，请使用 [GitHub Discussions](https://github.com/microsoft/onnxruntime/discussions)。

## 行为准则

本项目已采用 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。
更多信息请参阅[行为准则 FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
或联系 [opencode@microsoft.com](mailto:opencode@microsoft.com) 提出任何其他问题或意见。

## 许可证

本项目基于 [MIT License](LICENSE) 许可。
