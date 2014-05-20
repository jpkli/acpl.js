{
  "targets": [
    {
      "target_name": "addon",
      "cflags" : ["-fexceptions"],
      "sources": [
        "addon.cpp",
        ],
      "include_dirs": [
        "/usr/local/cuda-5.5/include/", "/home/kelvin/.node-gyp/0.10.26/deps/v8/src/"
      ],
      "libraries": [
        "-lcuda", "-lcudart", "-L/usr/local/cuda-5.5/lib64", "-L/home/kelvin/workspace/nodejs/jacl/lib", "-lacp"
      ]
    }
  ]
}
