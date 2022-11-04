#ifndef SHFL_FULLDTW_511
#define SHFL_FULLDTW_511

// 16 values per thread no shared memory
template <
    typename index_t,
    typename value_t> __global__
void shfl_FullDTW_511( // was DTW_fast_512_shuffle_kernel_no_shared_memory
    value_t * Subject,
    value_t * Dist,
    index_t num_entries,
    index_t num_features) {

    const index_t blid = blockIdx.x;
    const index_t thid = threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = blid*num_features;
    const index_t WARP_SIZE = 32;
    const index_t l = thid;

    //extern __shared__ value_t Subject_cache[];

    value_t penalty_left = INFINITY;
    value_t penalty_diag = 0;  // INFINITY;
    value_t penalty_here0 = INFINITY; // 0;
    value_t penalty_here1 = INFINITY; // 0;
    value_t penalty_here2 = INFINITY; // 0;
    value_t penalty_here3 = INFINITY; // 0;
    value_t penalty_here4 = INFINITY; // 0;
    value_t penalty_here5 = INFINITY; // 0;
    value_t penalty_here6 = INFINITY; // 0;
    value_t penalty_here7 = INFINITY; // 0;
    value_t penalty_here8 = INFINITY; // 0;
    value_t penalty_here9 = INFINITY; // 0;
    value_t penalty_here10 = INFINITY; // 0;
    value_t penalty_here11 = INFINITY; // 0;
    value_t penalty_here12 = INFINITY; // 0;
    value_t penalty_here13 = INFINITY; // 0;
    value_t penalty_here14 = INFINITY; // 0;
    value_t penalty_here15 = INFINITY; // 0;
    value_t penalty_temp0;
    value_t penalty_temp1;

        // Init shared memeory for right column
    //for (index_t l = thid; l < lane; l += blockDim.x)
    //    Subject_cache[l] = INFINITY;
    //__syncthreads();

    //index_t iter = 0;

    //for (index_t l = thid; l < lane/16; l += blockDim.x) {
    //    const index_t iter = l/WARP_SIZE;
        if (thid == 0) {
            penalty_left = INFINITY;
            penalty_diag = INFINITY;
            penalty_here0 = INFINITY; // 0;
            penalty_here1 = INFINITY; // 0;
            penalty_here2 = INFINITY; // 0;
            penalty_here3 = INFINITY; // 0;
            penalty_here4 = INFINITY; // 0;
            penalty_here5 = INFINITY; // 0;
            penalty_here6 = INFINITY; // 0;
            penalty_here7 = INFINITY; // 0;
            penalty_here8 = INFINITY; // 0;
            penalty_here9 = INFINITY; // 0;
            penalty_here10 = INFINITY; // 0;
            penalty_here11 = INFINITY; // 0;
            penalty_here12 = INFINITY; // 0;
            penalty_here12 = INFINITY; // 0;
            penalty_here13 = INFINITY; // 0;
            penalty_here14 = INFINITY; // 0;
            penalty_here15 = INFINITY; // 0;
        }

        //const value_t subject_value = Subject[base+l-1];
        const value_t subject_value0 = l == 0 ? 0 : Subject[base+16*l-1];
        const value_t subject_value1 = Subject[base+16*l-0];
        const value_t subject_value2 = Subject[base+16*l+1];
        const value_t subject_value3 = Subject[base+16*l+2];
        const value_t subject_value4 = Subject[base+16*l+3];
        const value_t subject_value5 = Subject[base+16*l+4];
        const value_t subject_value6 = Subject[base+16*l+5];
        const value_t subject_value7 = Subject[base+16*l+6];
        const value_t subject_value8 = Subject[base+16*l+7];
        const value_t subject_value9 = Subject[base+16*l+8];
        const value_t subject_value10 = Subject[base+16*l+9];
        const value_t subject_value11 = Subject[base+16*l+10];
        const value_t subject_value12 = Subject[base+16*l+11];
        const value_t subject_value13 = Subject[base+16*l+12];
        const value_t subject_value14 = Subject[base+16*l+13];
        const value_t subject_value15 = Subject[base+16*l+14];
        //if (blid == 0) Dist[2*l] = subject_value0;
        //if (blid == 0) Dist[2*l+1] = subject_value1;
        index_t counter = 1;
        value_t query_value = INFINITY;
        value_t new_query_value = cQuery[thid];
        if (thid == 0) query_value = new_query_value;
        if (thid == 0) penalty_here1 = 0; // (query_value - subject_value0)*(query_value - subject_value0);
        //penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here1, 1, 32);
        new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);
        //const index_t j = l;
        //if (blid == 0 && thid == 31 && iter == 0) Dist[0] = penalty_here0;
        //if (blid == 0 && thid == 31 && iter == 0) Dist[1] = penalty_here1;
        //if (blid == 0 && iter == 0) Dist[2*thid] = penalty_here0;
        //if (blid == 0 && iter == 0) Dist[2*thid+1] = penalty_here1;

        penalty_temp0 = penalty_here0;
        penalty_here0 = (query_value-subject_value0) * (query_value-subject_value0) + min(penalty_left, min(penalty_here0, penalty_diag));
        //if (i==2) penalty_temp1 = INFINITY; else penalty_temp1 = penalty_here1;  // -> move before main loop!!!
        penalty_temp1 = INFINITY;
        penalty_here1 = (query_value-subject_value1) * (query_value-subject_value1) + min(penalty_here0, min(penalty_here1, penalty_temp0));
        penalty_temp0 = penalty_here2;
        penalty_here2 = (query_value-subject_value2) * (query_value-subject_value2) + min(penalty_here1, min(penalty_here2, penalty_temp1));
        penalty_temp1 = penalty_here3;
        penalty_here3 = (query_value-subject_value3) * (query_value-subject_value3) + min(penalty_here2, min(penalty_here3, penalty_temp0));
        penalty_temp0 = penalty_here4;
        penalty_here4 = (query_value-subject_value4) * (query_value-subject_value4) + min(penalty_here3, min(penalty_here4, penalty_temp1));
        penalty_temp1 = penalty_here5;
        penalty_here5 = (query_value-subject_value5) * (query_value-subject_value5) + min(penalty_here4, min(penalty_here5, penalty_temp0));
        penalty_temp0 = penalty_here6;
        penalty_here6 = (query_value-subject_value6) * (query_value-subject_value6) + min(penalty_here5, min(penalty_here6, penalty_temp1));
        penalty_temp1 = penalty_here7;
        penalty_here7 = (query_value-subject_value7) * (query_value-subject_value7) + min(penalty_here6, min(penalty_here7, penalty_temp0));

        penalty_temp0 = penalty_here8;
        penalty_here8 = (query_value-subject_value8) * (query_value-subject_value8) + min(penalty_here7, min(penalty_here8, penalty_temp1));
        penalty_temp1 = penalty_here9;
        penalty_here9 = (query_value-subject_value9) * (query_value-subject_value9) + min(penalty_here8, min(penalty_here9, penalty_temp0));
        penalty_temp0 = penalty_here10;
        penalty_here10 = (query_value-subject_value10) * (query_value-subject_value10) + min(penalty_here9, min(penalty_here10, penalty_temp1));
        penalty_temp1 = penalty_here11;
        penalty_here11 = (query_value-subject_value11) * (query_value-subject_value11) + min(penalty_here10, min(penalty_here11, penalty_temp0));
        penalty_temp0 = penalty_here12;
        penalty_here12 = (query_value-subject_value12) * (query_value-subject_value12) + min(penalty_here11, min(penalty_here12, penalty_temp1));
        penalty_temp1 = penalty_here13;
        penalty_here13 = (query_value-subject_value13) * (query_value-subject_value13) + min(penalty_here12, min(penalty_here13, penalty_temp0));

        penalty_temp0 = penalty_here14;
        penalty_here14 = (query_value-subject_value14) * (query_value-subject_value14) + min(penalty_here13, min(penalty_here14, penalty_temp1));
        penalty_here15 = (query_value-subject_value15) * (query_value-subject_value15) + min(penalty_here14, min(penalty_here15, penalty_temp0));

        query_value = __shfl_up_sync(0xFFFFFFFF, query_value, 1, 32);
        if (thid == 0) query_value = new_query_value;
        new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);
        counter++;

        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here15, 1, 32);

        //if (iter && thid == 0) penalty_left = Subject_cache[2+1];
        if (thid == 0) penalty_left = INFINITY;

        for (index_t k = 3; k < lane+WARP_SIZE-1; k++) {
            const index_t i = k-l;
            //outside = k <= l || i >= lane;

            //const value_t residue = outside ? INFINITY : cQuery[i-1]-subject_value;
            //const value_t residue = outside ? INFINITY : query_value-subject_value;
            //if (thid == 0 && iter == 0 && k == 2) penalty_temp = INFINITY; else
            penalty_temp0 = penalty_here0;
            penalty_here0 = (query_value-subject_value0) * (query_value-subject_value0) + min(penalty_left, min(penalty_here0, penalty_diag));
            //if (i==2) penalty_temp1 = INFINITY; else penalty_temp1 = penalty_here1;  // -> move before main loop!!!
            penalty_temp1 = penalty_here1;
            penalty_here1 = (query_value-subject_value1) * (query_value-subject_value1) + min(penalty_here0, min(penalty_here1, penalty_temp0));
            penalty_temp0 = penalty_here2;
            penalty_here2 = (query_value-subject_value2) * (query_value-subject_value2) + min(penalty_here1, min(penalty_here2, penalty_temp1));
            penalty_temp1 = penalty_here3;
            penalty_here3 = (query_value-subject_value3) * (query_value-subject_value3) + min(penalty_here2, min(penalty_here3, penalty_temp0));
            penalty_temp0 = penalty_here4;
            penalty_here4 = (query_value-subject_value4) * (query_value-subject_value4) + min(penalty_here3, min(penalty_here4, penalty_temp1));
            penalty_temp1 = penalty_here5;
            penalty_here5 = (query_value-subject_value5) * (query_value-subject_value5) + min(penalty_here4, min(penalty_here5, penalty_temp0));
            penalty_temp0 = penalty_here6;
            penalty_here6 = (query_value-subject_value6) * (query_value-subject_value6) + min(penalty_here5, min(penalty_here6, penalty_temp1));
            penalty_temp1 = penalty_here7;
            penalty_here7 = (query_value-subject_value7) * (query_value-subject_value7) + min(penalty_here6, min(penalty_here7, penalty_temp0));

            penalty_temp0 = penalty_here8;
            penalty_here8 = (query_value-subject_value8) * (query_value-subject_value8) + min(penalty_here7, min(penalty_here8, penalty_temp1));
            penalty_temp1 = penalty_here9;
            penalty_here9 = (query_value-subject_value9) * (query_value-subject_value9) + min(penalty_here8, min(penalty_here9, penalty_temp0));
            penalty_temp0 = penalty_here10;
            penalty_here10 = (query_value-subject_value10) * (query_value-subject_value10) + min(penalty_here9, min(penalty_here10, penalty_temp1));
            penalty_temp1 = penalty_here11;
            penalty_here11 = (query_value-subject_value11) * (query_value-subject_value11) + min(penalty_here10, min(penalty_here11, penalty_temp0));
            penalty_temp0 = penalty_here12;
            penalty_here12 = (query_value-subject_value12) * (query_value-subject_value12) + min(penalty_here11, min(penalty_here12, penalty_temp1));
            penalty_temp1 = penalty_here13;
            penalty_here13 = (query_value-subject_value13) * (query_value-subject_value13) + min(penalty_here12, min(penalty_here13, penalty_temp0));

            penalty_temp0 = penalty_here14;
            penalty_here14 = (query_value-subject_value14) * (query_value-subject_value14) + min(penalty_here13, min(penalty_here14, penalty_temp1));
            penalty_here15 = (query_value-subject_value15) * (query_value-subject_value15) + min(penalty_here14, min(penalty_here15, penalty_temp0));

            //if (blid == 0 && thid == 31 && iter == 0) Dist[2*(k-1)] = penalty_here0;
            //if (blid == 0 && thid == 31 && iter == 0) Dist[2*(k-1)+1] = penalty_here1;
            //if (blid == 0 && iter == 0) Dist[64*(k-1)+2*thid] = penalty_here0;
            //if (blid == 0 && iter == 0) Dist[64*(k-1)+2*thid+1] = penalty_here1;

            //if (counter%32 == 0 && counter > 1) new_query_value = cQuery[i+2*thid-1];
            if (counter%32 == 0) new_query_value = cQuery[i+2*thid-1];
            query_value = __shfl_up_sync(0xFFFFFFFF, query_value, 1, 32);
            if (thid == 0) query_value = new_query_value;
            new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);
            //if (thid == 0) if (!outside) Dist[counter] = query_value; else Dist[counter] = 0;
            counter++;

            // save the right column
            //if (!outside && thid == 31 && iter < lane/WARP_SIZE-1) Subject_cache[i] = penalty_here; // TO DO: replace this by shhffles
            //if (iter < lane/WARP_SIZE-1 && thid == 31 && k>l) Subject_cache[i] = penalty_here15;

            // shuffle the penalty
            penalty_diag = penalty_left;
            penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here15, 1, 32);
            //if (thid == 0 && !outside) penalty_left = Subject_cache[i+1]; // TO DO: replace by shuffles
        //    if (iter > 0 && thid == 0 && k>l) penalty_left = Subject_cache[i+1];
            //if (iter && thid == 0) penalty_left = Subject_cache[i+1];
            //if (!iter && thid == 0) penalty_left = INFINITY;
            if (thid == 0) penalty_left = INFINITY;

            //if (thid == 0 && k>l) if (iter > 0) penalty_left = Subject_cache[i+1]; else penalty_left = INFINITY;
            //if (thid == 0 && k>l && iter > 0) penalty_left = Subject_cache[i+1];
        }
        penalty_temp0 = penalty_here0;
        penalty_here0 = (query_value-subject_value0) * (query_value-subject_value0) + min(penalty_left, min(penalty_here0, penalty_diag));
        penalty_temp1 = penalty_here1;
        penalty_here1 = (query_value-subject_value1) * (query_value-subject_value1) + min(penalty_here0, min(penalty_here1, penalty_temp0));
        penalty_temp0 = penalty_here2;
        penalty_here2 = (query_value-subject_value2) * (query_value-subject_value2) + min(penalty_here1, min(penalty_here2, penalty_temp1));
        penalty_temp1 = penalty_here3;
        penalty_here3 = (query_value-subject_value3) * (query_value-subject_value3) + min(penalty_here2, min(penalty_here3, penalty_temp0));
        penalty_temp0 = penalty_here4;
        penalty_here4 = (query_value-subject_value4) * (query_value-subject_value4) + min(penalty_here3, min(penalty_here4, penalty_temp1));
        penalty_temp1 = penalty_here5;
        penalty_here5 = (query_value-subject_value5) * (query_value-subject_value5) + min(penalty_here4, min(penalty_here5, penalty_temp0));
        penalty_temp0 = penalty_here6;
        penalty_here6 = (query_value-subject_value6) * (query_value-subject_value6) + min(penalty_here5, min(penalty_here6, penalty_temp1));
        penalty_temp1 = penalty_here7;
        penalty_here7 = (query_value-subject_value7) * (query_value-subject_value7) + min(penalty_here6, min(penalty_here7, penalty_temp0));

        penalty_temp0 = penalty_here8;
        penalty_here8 = (query_value-subject_value8) * (query_value-subject_value8) + min(penalty_here7, min(penalty_here8, penalty_temp1));
        penalty_temp1 = penalty_here9;
        penalty_here9 = (query_value-subject_value9) * (query_value-subject_value9) + min(penalty_here8, min(penalty_here9, penalty_temp0));
        penalty_temp0 = penalty_here10;
        penalty_here10 = (query_value-subject_value10) * (query_value-subject_value10) + min(penalty_here9, min(penalty_here10, penalty_temp1));
        penalty_temp1 = penalty_here11;
        penalty_here11 = (query_value-subject_value11) * (query_value-subject_value11) + min(penalty_here10, min(penalty_here11, penalty_temp0));
        penalty_temp0 = penalty_here12;
        penalty_here12 = (query_value-subject_value12) * (query_value-subject_value12) + min(penalty_here11, min(penalty_here12, penalty_temp1));
        penalty_temp1 = penalty_here13;
        penalty_here13 = (query_value-subject_value13) * (query_value-subject_value13) + min(penalty_here12, min(penalty_here13, penalty_temp0));

        penalty_temp0 = penalty_here14;
        penalty_here14 = (query_value-subject_value14) * (query_value-subject_value14) + min(penalty_here13, min(penalty_here14, penalty_temp1));
        penalty_here15 = (query_value-subject_value15) * (query_value-subject_value15) + min(penalty_here14, min(penalty_here15, penalty_temp0));

        //if (thid == 31 && iter < lane/WARP_SIZE-1) Subject_cache[lane+(iter+1)*WARP_SIZE-1-l] = penalty_here15;
        //iter++;
    
    if(thid == blockDim.x-1)  Dist[blid] = penalty_here15;
}



#endif
