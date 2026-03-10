#pragma once

template<typename T>
std::vector<T> to_vector(py::array_t<T>& array) {
    array = py::array::ensure(array, py::array::c_style);
    auto buf = array.request();
    T* ptr = static_cast<T*>(buf.ptr);
    auto size = buf.size;
    if ( size > 0 )
        return std::vector<T>(ptr, ptr + size);
    return std::vector<T>();
}

template<typename otherT, typename T>
std::vector<otherT> to_vector_cast(const py::array_t<T>& arrayIn) {
    auto array = py::array::ensure(arrayIn, py::array::c_style);
    auto buf = array.request();
    otherT* ptr = static_cast<otherT*>(buf.ptr);
    size_t size = buf.size * sizeof(T) / sizeof(otherT);
    if ( size > 0 )
        return std::vector<otherT>(ptr, ptr + size);
    return std::vector<otherT>();
}

using ShapeContainer = pybind11::detail::any_container<pybind11::ssize_t>;
template<typename T, typename otherT=T>
py::array_t<otherT> to_py_vector(std::vector<T>& data, ShapeContainer shape = ShapeContainer()) {
    if ( shape->empty() )
        shape = { data.size() };
    py::array_t<otherT> result(shape);
    auto buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);

    std::copy(data.begin(), data.end(), ptr);
    return result;
}
