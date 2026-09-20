#pragma
#include "memCache.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

namespace {

bool verbose()
{
    static const bool flag = std::getenv("GCDT_MEM_VERBOSE") != nullptr;
    return flag;
}

}  // namespace

void* MemCache::getMemByAbtrtSize(size_t size)
{
    auto it = m_freeBySize.find(size);
    if (it != m_freeBySize.end() && !it->second.empty())
    {
        void* memAddress = it->second.back();
        it->second.pop_back();
        m_cachedBytes -= size;
        m_liveBytes += size;
        m_peakLiveBytes = std::max(m_peakLiveBytes, m_liveBytes);
        if (verbose()) printf("reuse Mem: %zu\n", size);
        return memAddress;
    }
    return allocate(size);
}

void* MemCache::getMemBySize(size_t size)
{
    if (unordered_mp.find(size) != unordered_mp.end())
    {
        if (unordered_mp[size].empty())
        {
            return nullptr;
        }
        else
        {
            auto back = unordered_mp[size].back();
            unordered_mp[size].pop_back();
            return back;
        }
    }
    return nullptr;
}

void* MemCache::getPinnedPage()
{
    return m_pinnedPage;
}

void MemCache::releasePinnedPage(void* mem)
{
    m_pinnedPage = mem;
}

void MemCache::initAbtrtMem()
{
    // Intentionally empty: memory is allocated on demand (see the class comment).
}

void MemCache::report()
{
    printf("pool report: cached=%.2f MB live=%.2f MB peak=%.2f MB limit=%.2f MB\n",
           m_cachedBytes / 1048576.0, m_liveBytes / 1048576.0,
           m_peakLiveBytes / 1048576.0, m_cacheLimit / 1048576.0);
    for (auto& item : m_freeBySize)
    {
        if (!item.second.empty()) printf("  %zu bytes x %zu\n", item.first, item.second.size());
    }
}

void MemCache::releaseAbtrtMem(void* ptr, size_t size)
{
    if (ptr == nullptr) return;
    m_liveBytes = (m_liveBytes > size) ? m_liveBytes - size : 0;
    recycle(ptr, size);
}

void MemCache::releaseMem(void* ptr, size_t size)
{
    unordered_mp[size].push_back(ptr);
}

MemCache& getMemCacheRef()
{
    static MemCache cache;
    static bool first = true;
    if (first)
    {
        first = false;
        cache.initAbtrtMem();
    }
    return cache;
}

void MemCache::releaseCache()
{
    for (auto& item : unordered_mp)
    {
        for (auto ptr : item.second)
        {
            cudaFree(ptr);
        }
        item.second.clear();
    }
    for (auto& item : m_freeBySize)
    {
        for (auto ptr : item.second) cudaFree(ptr);
        item.second.clear();
    }
    m_freeBySize.clear();
    m_blockSizes.clear();
    m_cachedBytes = 0;
    m_liveBytes = 0;
    cudaFreeHost(m_pinnedPage);
    m_pinnedPage = nullptr;
}

void MemCache::setCacheLimit(size_t bytes)
{
    m_cacheLimit = bytes;
    // Drop cached blocks until the limit is respected.
    while (m_cachedBytes > m_cacheLimit && !m_freeBySize.empty())
    {
        auto it = m_freeBySize.begin();
        auto& blocks = it->second;
        if (blocks.empty())
        {
            m_freeBySize.erase(it);
            continue;
        }
        cudaFree(blocks.back());
        blocks.pop_back();
        m_cachedBytes -= it->first;
    }
}

size_t MemCache::cacheLimit() const { return m_cacheLimit; }
size_t MemCache::cachedBytes() const { return m_cachedBytes; }
size_t MemCache::liveBytes() const { return m_liveBytes; }
size_t MemCache::peakLiveBytes() const { return m_peakLiveBytes; }

void* MemCache::allocate(size_t size)
{
    void* memAddress = nullptr;
    cudaMalloc(&memAddress, size);
    if (verbose()) printf("new Mem: %zu\n", size);
    m_blockSizes[memAddress] = size;
    m_liveBytes += size;
    m_peakLiveBytes = std::max(m_peakLiveBytes, m_liveBytes);
    return memAddress;
}

void MemCache::recycle(void* ptr, size_t size)
{
    // Upstream releases buffers with a caller-guessed element count. Trust the
    // recorded allocation size when available: a wrong size would file the
    // block under the wrong key and hand out an undersized buffer later.
    auto recorded = m_blockSizes.find(ptr);
    if (recorded != m_blockSizes.end())
    {
        size = recorded->second;
        m_blockSizes.erase(recorded);
    }
    if (size <= m_cacheLimit && m_cachedBytes + size <= m_cacheLimit)
    {
        m_freeBySize[size].push_back(ptr);
        m_cachedBytes += size;
    }
    else
    {
        cudaFree(ptr);
    }
}
