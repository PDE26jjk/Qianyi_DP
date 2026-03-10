#include "InputCreator.h"

#include "HashTable.h"

#include<assert.h>


void InputCreator::readPoints
( 
std::string     inFilename, 
Point2HVec&     pointVec,
SegmentHVec&    constraintVec,
int             maxConstraintNum
)
{
    bool isBinary = ( 0 != inFilename.substr( inFilename.length() - 4, 4 ).compare( ".txt" ) ); 

    std::string vertFn = inFilename; 
    std::string consFn = inFilename; 

    Point2HVec  inPointVec;
    SegmentHVec inConstraintVec; 
    std::ifstream inFile;

    if ( isBinary ) 
    {
        std::cout << "Binary input file!" << std::endl; 

        // Vertices are in .vtx and constraints are in .cst files
        vertFn = inFilename; 
        vertFn.append( ".vtx" ); 
        inFile.open( vertFn.c_str(), std::ios::binary );
    }
    else
    {
        inFile.open( inFilename.c_str() );
    }

    if ( !inFile.is_open() )
    {
        std::cout << "Error opening input file: " << vertFn << " !!!" << std::endl;
        exit( 1 );
    }
    else
    {
        std::cout << "Reading from file... " << inFilename << std::endl;
    }

    if ( isBinary ) 
    {
        int pointNum, constraintNum; 

        // Get file size
        inFile.seekg( 0, inFile.end ); 

        const int fileSize = inFile.tellg(); 

        inFile.seekg( 0, inFile.beg ); 

        // Read pointNum
        inFile.read( (char*) &pointNum, sizeof( int ) );

        // Detect whether numbers are in float or double
        const int bufferSize = fileSize - sizeof( int ); 

        if ( 0 != bufferSize % pointNum ) 
        {
            std::cout << "Invalid input file format! Wrong file size: " << bufferSize << std::endl; 
            exit( -1 ); 
        }

        if ( bufferSize / pointNum == 8 ) 
        {
            // Float
            float* buffer = new float[ pointNum * 2 ]; 

            inFile.read( ( char * ) buffer, pointNum * 2 * sizeof( float ) );

            for ( int i = 0; i < pointNum * 2; i += 2 ) 
            {
                Point2 point = { buffer[ i ], buffer[ i + 1 ] }; 

                inPointVec.push_back( point ); 
            }

            delete [] buffer; 
        } 
        else if ( bufferSize / pointNum == 16 ) 
        {
            // Double
            double* buffer = new double[ pointNum * 2 ]; 

            inFile.read( ( char * ) buffer, pointNum * 2 * sizeof( double ) );

            for ( int i = 0; i < pointNum * 2; i += 2 ) 
            {
                Point2 point = { buffer[ i ], buffer[ i + 1 ] }; 

                inPointVec.push_back( point ); 
            }

            delete [] buffer; 
        }
        else
        {
            std::cout << "Unknown input number format! Size = " 
                << bufferSize / pointNum << std::endl; 
            exit( -1 ); 
        }

        inFile.close(); 
        
        consFn.append( ".cst" ); 
        inFile.open( consFn.c_str(), std::ios::binary );
    
        if ( inFile.is_open() )
        {
            inFile.read( (char*) &constraintNum, sizeof( int ) );

            inConstraintVec.resize( constraintNum ); 

            inFile.read( (char *) &inConstraintVec[0], constraintNum * 2 * sizeof( int ) ); 

            inFile.close(); 
        }
    } else
    {
        int pointNum, constraintNum, id;
        Point2 pt; 

        inFile >> pointNum; 

        for ( int i = 0; i < pointNum; ++i ) 
        {
            inFile >> id >> pt._p[0] >> pt._p[1]; 

            inPointVec.push_back( pt ); 
        }

		inFile.close(); 

		consFn = inFilename.substr(0, inFilename.length() - 4); 
        consFn.append( ".cst" ); 
        inFile.open( consFn.c_str() );

		inFile >> constraintNum; 

		Segment seg; 

        for ( int i = 0; i < constraintNum; ++i ) 
        {
            inFile >> id >> seg._v[0] >> seg._v[1]; 

            inConstraintVec.push_back( seg ); 
        }

        inFile.close(); 
        //int pointNum, constraintNum;
        //Point2 pt; 

        //inFile >> pointNum >> constraintNum; 

        //for ( int i = 0; i < pointNum; ++i ) 
        //{
        //    inFile >> pt._p[0] >> pt._p[1]; 

        //    inPointVec.push_back( pt ); 
        //}

        //Segment seg; 

        //for ( int i = 0; i < constraintNum; ++i ) 
        //{
        //    inFile >> seg._v[0] >> seg._v[1]; 

        //    inConstraintVec.push_back( seg ); 
        //}

        //inFile.close(); 
    }

    std::cout << "Number of points:      " << inPointVec.size() << std::endl; 
    std::cout << "Number of constraints: " << inConstraintVec.size() << std::endl; 

	if (maxConstraintNum == -1) 
		maxConstraintNum = inConstraintVec.size(); 

	printf("maxConstraintNum = %d\n", maxConstraintNum); 

    ////
    // Remove duplicates
    ////
    HashPoint2 hashPoint2; 
    PointTable pointSet( inPointVec.size(), hashPoint2 ); 

    IntHVec pointMap( inPointVec.size() ); 

    //pointSet.summary(); 

    // Iterate input points
    for ( int ip = 0; ip < ( int ) inPointVec.size(); ++ip )
    {
        Point2 inPt = inPointVec[ ip ];
        int ptIdx; 

        // Check if point unique
        if ( !pointSet.get( inPt, &ptIdx ) )
        {
            pointVec.push_back( inPt );

            ptIdx = pointVec.size() - 1; 

            pointSet.insert( inPt, ptIdx );
        }

        pointMap[ ip ] = ptIdx; 
    }

    const int dupCount = inPointVec.size() - ( int ) pointVec.size();

    if ( dupCount > 0 )
        std::cout << dupCount << " duplicate points in input file!" << std::endl;

    for ( int i = 0; i < ( int ) inConstraintVec.size(); ++i ) 
    {
        const Segment  inC = inConstraintVec[ i ]; 
        const Segment newC = { pointMap[ inC._v[0] ], pointMap[ inC._v[1] ] }; 

        if ( newC._v[0] != newC._v[1] && constraintVec.size() < maxConstraintNum ) 
            constraintVec.push_back( newC ); 
    }

    const int dupConstraint = inConstraintVec.size() - constraintVec.size(); 

    if ( dupConstraint > 0 ) 
        std::cout << dupConstraint << " degenerate or ignored constraints in input file!" << std::endl; 
}  
