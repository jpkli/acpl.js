#include <stdlib.h>
#include <cstring>
#include <node_buffer.h>
#include <stdio.h>
//#define BUILDING_NODE_EXTENSION
#include <node.h>
#include "lib/acp.hpp"
using namespace std;
using namespace v8;


Handle<Value> TestThrust(const Arguments& args) {	
    HandleScope scope;
    acp::testThrust();
    return scope.Close(Number::New(0));
}

Handle<Value> TestCUDA(const Arguments& args) {
    HandleScope scope;

    if (!args[0]->IsArray()) {
      ThrowException(Exception::TypeError(String::New("Wrong arguments")));
      return scope.Close(Undefined());
    }

    Local<Array> arr1 = Local<Array>::Cast(args[0]);
    Local<Array> arr2 = Local<Array>::Cast(args[1]);

    int N = arr1->Length();
    float* h_A = (float*) malloc( sizeof(float) * N );
    float* h_B = (float*) malloc( sizeof(float) * arr2->Length() );

    for(int i = 0; i < N; ++i)
    {
      h_A[i] = (float) arr1->Get(i)->NumberValue();
      h_B[i] = (float) arr2->Get(i)->NumberValue();
    };

    acp::testCUDA(h_A, h_B, N);

    Local<Array> result_array = Array::New(N);
    for(int i = 0; i < N; ++i)
    {
      result_array->Set(i, Number::New(h_A[i]));
    };

    return scope.Close(result_array);

}

Handle<Value> TestBuffer(const Arguments& args) { 
    HandleScope scope;
    Local<Object>  array1  = args[0]->ToObject();
    Handle<Object> buffer1 = array1->Get(String::New("buffer"))->ToObject();
    unsigned int   offset1 = array1->Get(String::New("byteOffset"))->Uint32Value();
    //int            length = array->Get(String::New("byteLength"))->Uint32Value();
    int            N = array1->Get(String::New("length"))->Uint32Value();
    //float *hA = NULL, *hB = NULL;

    Local<Object>  array2  = args[1]->ToObject();
    Handle<Object> buffer2 = array2->Get(String::New("buffer"))->ToObject();
    unsigned int   offset2 = array2->Get(String::New("byteOffset"))->Uint32Value();

    float *hA = (float*) &((char*) buffer1->GetIndexedPropertiesExternalArrayData())[offset1];
    float *hB = (float*) &((char*) buffer2->GetIndexedPropertiesExternalArrayData())[offset2];
    //hB = (float*) malloc( sizeof(float) * N );

    //memcpy(hB, hA, N);

    int test  = args[2]->Int32Value();
    if( test == 1) {
        acp::transform(hA, hB, N, "+");
        return scope.Close(array1);
    } else if( test == 2 ) {
        float result = acp::reduce(hA, N, "+");
        return scope.Close(Number::New(result));
    } else if( test == 3 ) {
        acp::sort(hA, N);
        return scope.Close(array1);
    } else {
        acp::scan(hA, N);
        return scope.Close(array1);
    }

    //Local<ArrayBuffer> sortedBuffer = ArrayBuffer::New(N*sizeof(float));
    //Local<Object> sortedArray = Object::New();
    //Local<Float32Array> sortedArray = Float32Array::New(sortedBuffer, 0, N);
    
    //Local<Object> arrayBuffer = Shell::CreateExternalArrayBuffer(N*sizeof(float));

    //Local<Object> sarray = Object::New();
    //buffer->SetIndexedPropertiesToExternalArrayData((char*) hB, kExternalUnsignedByteArray, N*sizeof(float));

   // array->Set(Symbols::buffer(isolate),buffer,ReadOnly);
/*
    Local<Object> sortedArray = Shell::CreateExternalArray(Arguments::New(), Float32Array, sizeof(float));
    Local<Object> buffer = sortedArray->Get(String::New("buffer"))->ToObject();
    buffer->SetIndexedPropertiesToExternalArrayData((char*) hB, kExternalUnsignedByteArray, N*sizeof(float));
*/
}

Handle<Value> Sort(const Arguments& args) { 
    HandleScope scope;
    Local<Array> iArray = Local<Array>::Cast(args[0]);
    int N = iArray->Length();
    float *h_iarray = (float*) malloc( sizeof(float) * N );

    for( int i = 0; i < N; ++i){
      h_iarray[i] = (float)(iArray->Get(i)->NumberValue());
    }

    acp::sort(h_iarray, N);

    Local<Array> oArray = Array::New(N);
    for( int i = 0; i < N; ++i){
      oArray->Set(i, Number::New(h_iarray[i]));
    }

    return scope.Close(oArray);
}

Handle<Value> Transform(const Arguments& args) {
    HandleScope scope;

    if (args.Length() < 3) {
      ThrowException(Exception::TypeError(String::New("Wrong number of arguments")));
      return scope.Close(Undefined());
    }

    if (!args[0]->IsArray() || !args[1]->IsArray() || !args[2]->IsString() ) {
      ThrowException(Exception::TypeError(String::New("Wrong arguments")));
      return scope.Close(Undefined());
    }

    Local<Array> arr1 = Local<Array>::Cast(args[0]);
    Local<Array> arr2 = Local<Array>::Cast(args[1]);
    String::AsciiValue opstr(args[0]->ToString());


    char *op = (char *) malloc(opstr.length() + 1);
    strcpy(op, *opstr); 

    int N = arr1->Length();
    float* h_A = (float*) malloc( sizeof(float) * N );
    float* h_B = (float*) malloc( sizeof(float) * arr2->Length() );

    for(int i = 0; i < N; ++i)
    {
      h_A[i] = (float) arr1->Get(i)->NumberValue();
      h_B[i] = (float) arr2->Get(i)->NumberValue();
    };

    acp::transform(h_A, h_B, N, op);

    Local<Array> result_array = Array::New(N);
    for(int i = 0; i < N; ++i)
    {
      result_array->Set(i, Number::New(h_A[i]));
    };

    return scope.Close(result_array);

}

Handle<Value> Reduce(const Arguments& args) {
    HandleScope scope;

    if (args.Length() < 2) {
      ThrowException(Exception::TypeError(String::New("Wrong number of arguments")));
      return scope.Close(Undefined());
    }

    if (!args[0]->IsArray() || !args[1]->IsString() ) {
      ThrowException(Exception::TypeError(String::New("Wrong arguments")));
      return scope.Close(Undefined());
    }

    Local<Array> arr1 = Local<Array>::Cast(args[0]);
    String::AsciiValue opstr(args[1]->ToString());

    char *op = (char *) malloc(opstr.length() + 1);
    strcpy(op, *opstr); 

    int N = arr1->Length();
    float* h_A = (float*) malloc( sizeof(float) * N );

    for(int i = 0; i < N; ++i)
    {
      h_A[i] = (float) arr1->Get(i)->NumberValue();
    };

    float result = acp::reduce(h_A, N, op);

    return scope.Close(Number::New(result));

}

Handle<Value> Scan(const Arguments& args) { 
    HandleScope scope;
    Local<Array> iArray = Local<Array>::Cast(args[0]);
    int N = iArray->Length();
    float *h_iarray = (float*) malloc( sizeof(float) * N );

    for( int i = 0; i < N; ++i){
      h_iarray[i] = (float)(iArray->Get(i)->NumberValue());
    }

    acp::scan(h_iarray, N);

    Local<Array> oArray = Array::New(N);
    for( int i = 0; i < N; ++i){
      oArray->Set(i, Number::New(h_iarray[i]));
    }

    return scope.Close(oArray);
}


void Init(Handle<Object> exports) {

    exports->Set(String::NewSymbol("testCUDA"), FunctionTemplate::New(TestCUDA)->GetFunction());
    exports->Set(String::NewSymbol("testBuffer"), FunctionTemplate::New(TestBuffer)->GetFunction());
    exports->Set(String::NewSymbol("testThrust"), FunctionTemplate::New(TestThrust)->GetFunction());

    exports->Set(String::NewSymbol("transform"), FunctionTemplate::New(Transform)->GetFunction());
    exports->Set(String::NewSymbol("sort"), FunctionTemplate::New(Sort)->GetFunction());
    exports->Set(String::NewSymbol("reduce"), FunctionTemplate::New(Reduce)->GetFunction());
    exports->Set(String::NewSymbol("scan"), FunctionTemplate::New(Scan)->GetFunction());
}

NODE_MODULE(addon, Init);
