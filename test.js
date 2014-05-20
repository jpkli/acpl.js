
var acp = require("./build/Release/addon");

//console.log("addon:", acp);

var N = process.argv[2];
var test = process.argv[3];


var jsFunction = {};

jsFunction.transform = function(A,B){
	var C = [];
	for(var i = 0; i < A.length; i++){
        C[i] = A[i] + B[i];
	}
	return C;
}

jsFunction.reduce = function(A){
	var c = 0;
	for(var i = 0; i < A.length; i++){
        c = c + A[i];
	}
	return c;
}

jsFunction.scan = function(A){
	var C = [];
	for(var i = 0; i < A.length; i++){
		var tmp_sum = 0;
		for(var j = 0; j<i; j++)
        	tmp_sum += A[i]; 

        C[i] = tmp_sum;
	}
	return C;
}


/*
var js = [];
var acp = [];
var acp_b = [];
var num_elements = [];

var results = [];
*/
//num_elements.push(N);
//var N = 1024 * 1024 * 16 ; //~16M elements

/*
for(var i = 0; i < 32; i++)
    console.log(rs[i]);
*/

var arr1 = [], arr2 = [];
for(var i = 0; i < N; i++){
   arr1.push(Math.random()+1000+1.234);
   arr2.push(Math.random()*1000+1.23);
}

var arr3 = arr1, arr4 = arr2, arr5 = arr1, arr6=arr2


if(test == "v8"){
 var start = Date.now();
 //var rs1 = jsFunction.transform(arr1, arr2);

//var rs1 = jsFunction.reduce(arr1);
var rs1 = arr1.sort(function(a,b){return a-b});
//var rs1 = jsFunction.scan(arr1);


 var end = Date.now();
 //console.log( ["V8:", N, "elements: ", end - start, "ms."].join(" "));
 process.stdout.write( end - start + ",");
}

//js.push(end-start);
//console.log(arr1);
//console.log(rs2);


if(test=="acp"){
	var start = Date.now();
	//var rs2 = acp.transform(arr3, arr4, "+");
	//var rs2 = acp.reduce(arr3, "+");
	var rs2 = acp.sort(arr3);

	//var rs2 = acp.scan(arr3)


	var end = Date.now();
	//console.log( ["ACP:", N, "elements: ", end - start, "ms."].join(" "));
	//acp.push(end-start);
    process.stdout.write( end - start + ",");
}

//console.log(arr1);

//var rs2 = acp.scan(arr1);
//console.log(rs2);

if(test=="acpb"){

	var ab1 = new ArrayBuffer(N*4);
	var av1 = new Float32Array(ab1);

	var ab2 = new ArrayBuffer(N*4);
	var av2 = new Float32Array(ab2);
	for(var i = 0; i < N; i++){
	   av1[i] = arr5[i];
	   av2[i] = arr6[i];
	}
	var start = Date.now();
	var rs3 = acp.testBuffer(av1, av2, 3);

	var end = Date.now();
	//console.log( ["ACP*:", N, "elements: ", end - start, "ms."].join(" "));
	process.stdout.write( end - start + ",");
}

//acp_b.push(end-start);

//var tr = acp.transform(arr1, arr5, "+");
//results.push( acp.transform(arr3, tr, "+") );

//console.log(rs3);

/*
console.log(num_elements.join(","))
console.log(js.join(","));
console.log(acp.join(","));
console.log(acp_b.join(","));
*/
//console.log(results.length);