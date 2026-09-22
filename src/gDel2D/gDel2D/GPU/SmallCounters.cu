#include "SmallCounters.h"

#include "HostToKernel.h"

void SmallCounters::init( int size, int capacity )
{
    _data.assign( capacity, 0 ); 

    _offset = 0; 
    _size   = size; ; 

}

SmallCounters::~SmallCounters()
{
    free(); 
}

void SmallCounters::free() 
{
    _data.free();
}

void SmallCounters::renew() 
{
    if ( _data.size() == 0 ) 
    {
        // Local modification (see README.txt): report it instead of exiting the
        // host process.
        throw std::runtime_error( "gDel2D: SmallCounters::renew() called before init()" ); 
    }

    _offset += _size;  

    if ( _offset + _size > _data.capacity() ) 
    {
        _offset = 0; 
        _data.fill( 0 ); 
    }
}

int* SmallCounters::ptr() 
{
    return toKernelPtr( _data ) + _offset; 
}

int SmallCounters::operator[]( int idx ) const
{
    assert( idx < _size ); 

    return _data[ _offset + idx ]; 
}
