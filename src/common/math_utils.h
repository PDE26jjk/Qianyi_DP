#pragma once

#include <cstdint>

static uint32_t mod_pow(uint32_t a, uint32_t b, uint32_t mod) {
    uint64_t res = 1;
    uint64_t base = a % mod;
    while ( b ) {
        if ( b & 1 ) res = (res * base) % mod;
        base = (base * base) % mod;
        b >>= 1;
    }
    return (uint32_t)res;
}

// Deterministic Miller-Rabin test (for 32-bit integers)
static bool is_prime(uint32_t n) {
    if ( n < 2 ) return false;
    if ( n == 2 || n == 3 ) return true;
    if ( n % 2 == 0 || n % 3 == 0 ) return false;

    const uint32_t bases[] = { 2, 7, 61 };

    uint32_t d = n - 1;
    int r = 0;
    while ( d % 2 == 0 ) {
        d >>= 1;
        r++;
    }

    for ( uint32_t a : bases ) {
        if ( a % n == 0 ) continue;
        uint32_t x = mod_pow(a, d, n);
        if ( x == 1 || x == n - 1 ) continue;

        bool composite = true;
        for ( int i = 0; i < r - 1; i++ ) {
            x = (uint64_t)x * x % n;
            if ( x == n - 1 ) {
                composite = false;
                break;
            }
        }
        if ( composite ) return false;
    }
    return true;
}

static uint32_t next_prime(uint32_t x) {
    if ( x < 2 ) return 2;
    
    uint32_t n = (x % 2 == 0) ? x + 1 : x + 2;
    while ( !is_prime(n) ) {
        n += 2;
    }
    return n;
}
