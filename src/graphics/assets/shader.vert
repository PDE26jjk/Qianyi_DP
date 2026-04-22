#version 450
// 输入：对应 C++ 中的 WindingVertex 结构体
layout(location = 0) in vec2 inPos;
layout(location = 1) in int inWeight;
// 推送常量：对应 C++ 中的 float[2] {width, height}
layout(push_constant) uniform PushConstants {
    vec2 resolution; // .x = width, .y = height
} pc;
// 输出给片段着色器
// "flat" 关键字非常重要：告诉 GPU 不要插值，保持整数值传递
layout(location = 0) flat out int outWeight;
void main() {
    // 1. 归一化坐标 (0.0 到 1.0)
    // 注意：如果您的坐标是像素坐标，除以分辨率
    vec2 normalizedPos = inPos / pc.resolution;
    // vec2 normalizedPos = inPos;
    // 2. 转换为 NDC (-1.0 到 1.0)
    // Vulkan NDC: X 向右, Y 向下
    // (x/width) * 2 - 1
    vec2 ndc = normalizedPos * 2.0 - 1.0;
    // 3. 输出位置
    // gl_Position 是内置变量
    gl_Position = vec4(ndc.x, ndc.y, 0.0, 1.0);
    // 4. 传递权重
    outWeight = inWeight;
}