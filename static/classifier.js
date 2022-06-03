//========================================================================
// Drag and drop image handling
//========================================================================


var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result2");
var loader = document.getElementById("loader");
var model_first = undefined;
var model_second = undefined;
var model_third  = undefined;
//========================================================================
// Main button events
//========================================================================


async function initialize() {
    model_first = await tf.loadLayersModel('/weights_first/model.json');
    model_second = await tf.loadLayersModel('/weights_second/model.json');
    model_third = await tf.loadLayersModel('/weights_third/model.json');
}

async function predict() {
  // action for the submit button
  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select an image before submit.");
    return;
  }

  let tensorImg = tf.browser.fromPixels(imagePreview)
	.resizeNearestNeighbor([224,224]) // change the image size here
	.toFloat()
	.div(tf.scalar(255.0))
	.expandDims();
	;

  console.log('Input Image shape: ', tensorImg.shape);
  prediction = await model_first.predict(tensorImg).data();

  if (prediction[0] > 0.66) {
      console.log ("I think it's a brain image");
      predResult.innerHTML = prediction[0];
      prediction_middle = await model_second.predict(tensorImg).data();
      if (prediction_middle[0] > 0.5) {
        predResult.innerHTML = prediction_middle[0];
        console.log ("I think it's a tumor brain image");

        let tensorImg_1 = tf.browser.fromPixels(imagePreview)
      	.resizeNearestNeighbor([256,256]) // change the image size here
	      .mean(2)
        .toFloat()
	      .div(tf.scalar(255.0))
        .expandDims(0)
	      .expandDims(-1);
	      
        prediction_last = await model_third.predict(tensorImg_1).data();
        
        	
	// convert typed array to a javascript array
	var preds = Array.from(prediction_last); // JS Code with js array
		
	
	// threshold the predictions
	var i;
	var num;
	for (i = 0; i < preds.length; i++) { 
		
		num = preds[i];
		
		if (num < 0.7) {   // <-- Set the threshold here
			preds[i] = 0;
			
		} else {
			preds[i] = 255;
			
		}
		
	}
		
		
	// convert js array to a tensor
	pred_tensor = tf.tensor1d(preds, 'int32');
	
	
	// reshape the pred tensor
	pred_tensor = pred_tensor.reshape([256,256,1]);
	
	// resize pred_tensor
	pred_tensor = pred_tensor.resizeNearestNeighbor([orig_image.shape[0], orig_image.shape[1]]);
	
	// reshape the input image tensor
	//input_img_tensor = tensor.reshape([128,128,3]);
	
	// append the tensor to the input image to create a 4th alpha channel --> shape [128,128,4]
	rgba_tensor = tf.concat([orig_image, pred_tensor], axis=-1);
	
	
	// resize all images. rgb_tensor is the segmented image.
	rgba_tensor = rgba_tensor.resizeNearestNeighbor([256, 256]);
	orig_image = orig_image.resizeNearestNeighbor([256, 256]);
	color_image = color_image.resizeNearestNeighbor([256, 256]);
	
	
	
	
	// Convert the tensor to an image 
	
	
	// Method 1: Canvas - Display pred image using tf.toPixels
	//=====================================================
	
	var canvas2 = document.getElementById("myCanvas2");
	var canvas3 = document.getElementById("myCanvas3");
	var canvas4 = document.getElementById("myCanvas4");

  tf.toPixels(rgba_tensor, canvas2);
	tf.toPixels(orig_image, canvas3);
	tf.toPixels(color_image, canvas4);
      }
      else{
        predResult.innerHTML = prediction_middle[0];
        console.log ("I think it's a normal brain image");
      }


  } else if (prediction[0]> 0.33) {
    //   console.log ( "I think it's a brain image");
      predResult.innerHTML = prediction[0];

  } else if (prediction[0] < 0.33) {
    // console.log ("I think it's a non-brain image");
    predResult.innerHTML = prediction[0];
  } else {
      predResult.innerHTML = "This is Something else";
      predResult.innerHTML = prediction[0];
  }
  show(predResult)

}

function clearImage() {
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  imagePreview.src = "";
  imageDisplay.src = "";
  predResult.innerHTML = "";

  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  show(uploadCaption);

  imageDisplay.classList.remove("loading");
}

function previewFile(file) {
  // show the preview of the image
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = URL.createObjectURL(file);

    show(imagePreview);
    hide(uploadCaption);

    // reset
    predResult.innerHTML = "";
    imageDisplay.classList.remove("loading");

    displayImage(reader.result, "image-display");
  };
}

//========================================================================
// Helper functions
//========================================================================

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}

initialize();