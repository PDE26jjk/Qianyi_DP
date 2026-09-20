#pragma once
#include <cstddef>
#include <list>
#include <unordered_map>
#include <vector>

// Lazy, size-exact device memory cache.
//
// The original implementation preallocated fixed buckets (about 3.2 GB in
// total) during initAbtrtMem(), and the demo entry point called initAbtrtMem()
// twice, so a run reserved roughly 6.4 GB before doing any work. That cannot
// run on low-end GPUs together with another GPU client.
//
// This version never preallocates: blocks are cudaMalloc'ed on demand at the
// exact requested size (no rounding up to a coarse bucket, which used to turn
// a 48 MB request into a 512 MB allocation) and recycled through a per-size
class MemCache
{
public:
    void* getMemByAbtrtSize(size_t size);
    void* getMemBySize(size_t size);

    void releaseAbtrtMem(void* ptr, size_t size);
    void releaseMem(void* ptr, size_t size);
    void initAbtrtMem();
    void releaseCache();

    void* getPinnedPage();
    void releasePinnedPage(void* mem);
    void report();

    // Maximum amount of freed memory kept for reuse (default 256 MB).
    void setCacheLimit(size_t bytes);
    size_t cacheLimit() const;
    size_t cachedBytes() const;
    size_t liveBytes() const;
    size_t peakLiveBytes() const;

private:
    std::unordered_map<size_t, std::list<void*>> unordered_mp;
    std::unordered_map<size_t, std::vector<void*>> m_freeBySize;
    std::unordered_map<void*, size_t> m_blockSizes;  // exact size of every live block
    size_t m_cacheLimit = 256ull * 1024ull * 1024ull;
    size_t m_cachedBytes = 0;
    size_t m_liveBytes = 0;
    size_t m_peakLiveBytes = 0;
    void* m_pinnedPage{ nullptr };

    void* allocate(size_t size);
    void recycle(void* ptr, size_t size);
};

MemCache& getMemCacheRef();
