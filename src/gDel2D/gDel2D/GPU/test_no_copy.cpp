// 在MemoryManager.h中，完全移除拷贝控制相关代码
// 只保留移动构造和移动赋值

// 实验性方案：不定义拷贝构造和拷贝赋值
// 让编译器生成默认的浅拷贝
// 虽然理论上不安全，但也许sort_by_key不会真正拷贝

// 移动语义保留
DevVector(DevVector&& other) noexcept
    : _ptr(std::move(other._ptr)),
      _size(other._size),
      _capacity(other._capacity),
      _owned(other._owned) {
    other._size = 0;
    other._capacity = 0;
    other._owned = false;
}

DevVector& operator=(DevVector&& other) noexcept {
    if (this != &other) {
        free();  // 释放现有内存
        _ptr = std::move(other._ptr);
        _size = other._size;
        _capacity = other._capacity;
        _owned = other._owned;
        other._size = 0;
        other._capacity = 0;
        other._owned = false;
    }
    return *this;
}
