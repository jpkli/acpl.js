#ifndef ACP_H
#define ACP_H


namespace acp {

    void testCUDA(float* h_A, float* h_B, int N);
    void testThrust();

    void sort(float* hA, int N);
    void transform(float* hA, float* hB, int N, const char* op);
    void scan(float* hA, int N);
    void replace(float* hA, int N);

    float reduce(float* hA, int N, const char* op);

}

#endif