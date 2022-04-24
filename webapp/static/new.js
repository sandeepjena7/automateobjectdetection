var models={
    Detectron2:['faster_rcnn_R_50_C4_1x','faster_rcnn_R_50_DC5_1x-','faster_rcnn_R_50_FPN_1x','faster_rcnn_R_50_C4_3x','faster_rcnn_R_50_DC5_3x','faster_rcnn_R_50_FPN_3x','faster_rcnn_R_101_C4_3x','faster_rcnn_R_101_DC5_3x','faster_rcnn_R_101_FPN_3x','retinanet_R_50_FPN_1x','retinanet_R_50_FPN_3x','retinanet_R_101_FPN_3x','rpn_R_50_C4_1x','rpn_R_50_FPN_1x','fast_rcnn_R_50_FPN_1x'],
    TF1:['ssd_mobilenet_v1_coco','ssd_inception_v2_coco','faster_rcnn_inception_v2_coco','faster_rcnn_resnet50_coco','faster_rcnn_resnet101_coco'],
    TF2:['CenterNet HourGlass104 512x512','CenterNet HourGlass104 1024x1024','EfficientDet D0 512x512','EfficientDet D1 640x640','EfficientDet D1 640x640','EfficientDet D2 768x768','EfficientDet D3 896x896','EfficientDet D4 1024x1024','SSD MobileNet V1 FPN 640x640','SSD ResNet50 V1 FPN 640x640 (RetinaNet50)','Faster R-CNN ResNet50 V1 640x640','Faster R-CNN ResNet101 V1 640x640','Faster R-CNN Inception ResNet V2 640x640'],
    YOLOV5:['YOLOv5n','YOLOv5s','YOLOv5m','YOLOv5l','YOLOv5x']
}

// var main = $('#frameworks');
// var sub = $('#sub_menu');

var main = document.getElementById('frameworks');
var sub = document.getElementById('sub_menu');


$(document).ready(function() {
  $('#new_img5').click(function() {
      var value = $("input[type=radio][name=radio]:checked").val();
      if (value == 'CUST') {
        $("#img3_a").css("visibility", "visible");
        $("#img3_b").css("visibility", "hidden");
        console.log("We are working with custom model")
      }
      else {
        $("#img3_a").css("visibility", "hidden");
        $("#img3_b").css("visibility", "visible");
        console.log("We are working with Pretrained model")
      }
  })
});


main.addEventListener('change', function(){

    var selected_option = models[this.value];
    $("#img3").css("visibility", "visible");
    console.log("frame is",this.value)
    while(sub.options.length > 0){
        sub.options.remove(0)
    }

Array.from(selected_option).forEach(function(ele){
    let option =new Option(ele,ele);
    sub.appendChild(option);
})


});

window.onload = () => {
    $("#sendbutton").click(() => {
      var imagebox = $("#imagebox");
      var link = $("#link");
      var input = $("#imageinput")[0];
      var model_file = $("#model_file")[0];
      var radio = $("input[type=radio][name=radio]:checked").val();

      
      console.log("Filename files [0]",model_file.files[0])
      console.log("Filename2 files [0]",model_file.files[1])

      console.log("is it custom or pretrained :",radio)
      console.log("Files in labelmap are",model_file.files);
      console.log(input.files);
      console.log("Framework chosen is",main.value);
      console.log("Model choosen for that framework is",sub.value);

      if (input.files && input.files[0]) {
        let formData = new FormData();
        formData.append("video", input.files[0]);
        formData.append("frame_work",main.value);
        formData.append("model_type",radio)
        formData.append("model",sub.value);
        formData.append("model_files1",model_file.files[0])
        formData.append("model_files2",model_file.files[1])
        formData.append("model_files3",model_file.files[2])
        formData.append("model_files4",model_file.files[3])

        console.log("formData is",formData);
        $.ajax({
          url: "/detect", // fix this to your liking
          type: "POST",
          data: formData,
          cache: false,
          processData: false,
          contentType: false,
          error: function (data) {
            console.log("upload error", data);
            console.log(data.getAllResponseHeaders());
          },
          success: function (data) {
            console.log(data);
            // bytestring = data["status"];
            // image = bytestring.split("'")[1];
            $("#link").css("visibility", "visible");
            $("#download").attr("href", "static/" + data[1]);
// new changes            
            $(".res-part").html("");
//            $(".res-part2").html("");
//            var imageData = data[1].image; 
//            $(".res-part2").append("<img class='resp-img' src='data:image/jpeg;base64," + imageData + "' alt='' />");
            $(".res-part").html("<pre>" + JSON.stringify(data[0], undefined, 2) + "</pre>");	
            $(".res-part").html("<pre>" + JSON.stringify(data[0], undefined, 2) + "</pre>");	
//            $("#loading").hide();	 
// new changes  
            console.log(data);
          },
        });
      }
    });
  };
  
  function readUrl(input) {
    imagebox = $("#imagebox");
    console.log("image box is",imagebox);
    console.log("input is",input)
    console.log("evoked readUrl");
    console.log(input.files)
    console.log("input files [0]",input.files[0])
    if (input.files && input.files[0]) {
      let reader = new FileReader();
      reader.onload = function (e) {
        console.log(e.target);
  
        imagebox.attr("src", e.target.result);
        //   imagebox.height(500);
        //   imagebox.width(800);
      };
      reader.readAsDataURL(input.files[0]);
    }
  }
  
  function readfiles(filenames) {

    console.log("input is",filenames)
    console.log("length is",filenames.files.length)
    console.log("input files [0]",filenames.files[0])
    if (filenames.files && filenames.files[0]) {
      let reader = new FileReader();
      reader.onload = function (e) 
      {
      alert(filenames.files.length," files are uploaded")
      }
    }
    else {
      alert("Please select files first")
    }
    };
  