#version 450
// 来自顶点着色器的输入 (必须也是 flat)
layout(location = 0) flat in int inWeight;
// 输出颜色 (对应 VK_FORMAT_R32_SFLOAT)
// 即使格式是 R32，这里仍然输出 vec4，但只有 R 通道会被写入
layout(location = 0) out vec4 outColor;
void main() {
	// 将整数权重转换为浮点数并输出
	// 如果 weight 是 1，输出 1.0；如果是 -1，输出 -1.0
	outColor = vec4(float(inWeight), 0.0, 0.0, 1.0);
}