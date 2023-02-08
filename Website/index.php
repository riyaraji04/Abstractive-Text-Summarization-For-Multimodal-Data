
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Abstractive Text Summarization For Multimodal Data</title>
    <link rel="stylesheet" type="text/css" href="style.css">

    <!-- <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">

    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>

    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script> -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
  <script src="https://kit.fontawesome.com/yourcode.js" crossorigin="anonymous"></script>

    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script> -->
<script type="text/javascript">





function myFunction() {
var checkBox = document.getElementById("myCheck");
var text = document.getElementById("text");
var image=document.getElementById("image");
if (checkBox.checked == true){
  if (document.getElementById("uploadi")){
    text.style.display = "block";
  }else if (document.getElementById("uploadt")) {
    image.style.display = "block";
  }
} else {
    text.style.display = "none";
}
}


function myOutput() {
var checkBox1 = document.getElementById("myCheck1");
var text = document.getElementById("output");
if (checkBox1.checked == true){
    text.style.display = "block";
} else {
    text.style.display = "none";
}
}



</script>





</head>
<body>
  <nav class="navbar navbar-inverse">
    <div class="container-fluid n">
      <div class="navbar-header">
        <ul class="nav navbar-nav navbar-right" style="margin-top:5px;">
        <li class="active" style="margin-left:3px"><a class="navbar-brand" href="#">LUMINIAIRE</a></li>
      </ul>
      </div>
      <ul class="nav navbar-nav">
        <!-- <li class="active"><a href="#"><span class="glyphicon glyphicon-home"></span> Home</a></li> -->
        <!-- <li><a href="about.html">About</a></li>
        <li><a href="#">Page 2</a></li> -->
      </ul>
      <ul class="nav navbar-nav navbar-right"  style="margin-top:5px;">
        <li class="active" style="margin-right:2px"><a href="#"><span class="glyphicon glyphicon-home"></span> Home</a></li>
        <li  class="active"><a href="about.html">About Us</a></li>

      </ul>
      <!-- <ul class="nav navbar-nav navbar-right">
        <li><a href="#"><span class="glyphicon glyphicon-user"></span> Sign Up</a></li>
        <li><a href="#"><span class="glyphicon glyphicon-log-in"></span> Login</a></li>
      </ul> -->
    </div>
  </nav>

    <div class="bgimage">
        <div class="container text-center headerset">
            <h1>Abstractive Text Summarizer <span class="glyphicon glyphicon-file"></span></h1>
            <h3>Now get summary for any long documents within minutes using Luminaire. Our website will help you understand the gist of  your lengthy documents in no time!</h3>
<a href="#section1" class="btn qwe"><h5>Get Started</h5></a>

        </div>
    </div>
    <br><br><br><br>

    <section class="container input text-center" id="section1">
        <h2>Select Input Type</h2>

        <div class="row">
            <script>
                function changeContent1() {
                        document.getElementById("show1").classList.remove("hide");
                        document.getElementById("show2").classList.add("hide");
                        document.getElementById("show3").classList.add("hide");
                        }

                        function changeContent2() {
                        document.getElementById("show1").classList.add("hide");
                        document.getElementById("show2").classList.remove("hide");
                        document.getElementById("show3").classList.add("hide");
                        }

                        function changeContent3() {
                        document.getElementById("show1").classList.add("hide");
                        document.getElementById("show2").classList.add("hide");
                        document.getElementById("show3").classList.remove("hide");
                        }
                        function go(){
                            location.href='text.php';
                        }
                        function go2(){
                          location.href='image.php'
                        }
                        function go3(){
                          location.href='video.php'
                        }
            </script>

           <div class="col-lg-4 col-md-4 col-sm-4 col-10 d-block m-auto" >
                <label class="custom-radio-btn" for="txt">
                <span class="label">Text </span>
                <input type="radio" name="sample" id="txt" onclick="go()" value="1" checked>
                <span class="checkmark"></span>
                </label>
            </div>
            <div class="col-lg-4 col-md-4 col-sm-4 col-10 d-block m-auto">
                <label class="custom-radio-btn" for="img">
                <span class="label">Image</span>
                <input type="radio" name="sample" id="img" onclick="go2();" value="2" checked>
                <span class="checkmark"></span>
                </label>

            </div>
            <div class="col-lg-4 col-md-4 col-sm-4 col-10 d-block m-auto">
                <label class="custom-radio-btn" for="vid">
                <span class="label">Video</span>
                <input type="radio" name="sample" id="vid" onclick="go3();" value="3" checked>
                <span class="checkmark"></span>
                </label>
            </div>
            <div id="show1" class="hide">
                <form id="uploadt" action="upload_txt.php" method="POST" enctype="multipart/form-data">
                    <input type="file" name="file" class="browse" style="margin-left: 90px;">
                    <button type="submit" name="submit" class="upload "style="margin-right:150px" >Upload Text</button>
                </form>
            </div>

            <div id="show2" class="hide">
                     <form id="uploadi" action="upload_img.php" method="POST" enctype="multipart/form-data" onsubmit="imgup()">
                         <input type="file" name="file" class="browse" style="margin-left:400px">
                         <button type="submit" name="submit" class="upload " style="margin-left:20px">Upload Image</button>
                     </form>
             </div>


            <div id="show3" class="hide">
                    <form id="uploadv" action="upload_video.php" method="POST" enctype="multipart/form-data">
                        <input type="file" name="file"class="browse" style="margin-left:800px" onclick="vidup">
                        <button type="submit" name="submit" class="upload " style="margin-left:800px">Upload Video</button>
                    </form>
            </div>

        </div>
        <script>

        // <?php
        //     $command = escapeshellcmd('python convtext.py');
        //     $output = shell_exec($command);
        //     echo $output;
        // ?>
// function imgup(){
//   <?PHP
//   shell_exec("python convtext.py -n 1");
// ?>
// }
// function vidup(){
//   <?PHP
//   echo shell_exec("python convideo.py");
// ?>
// }



  </script>

    </section>

<br><br><br><br>

<!--for the input box-->
    <section class="container extractedtext text-center" id="section2">
        <h2>Extracted Text</h2>
        <input type="checkbox" id="myCheck" onclick="myFunction()">
        <label for="myCheck"><h3>Check the box to View the text to be summarized</h3> </label>


        <div id="text" style="display:none">
            <p><iframe src="uploads/uploadedtext.txt" frameborder="0" height="400"
            width="95%"></iframe></p>
        </div>
        <!-- .......... -->
        <div id="image" style="display:none">
            <p><iframe src="uploads/uploadedimage.jpg" frameborder="0" height="400"
            width="95%"></iframe></p>
        </div>
        <div id="video" style="display:none">
            <p><iframe src="uploads/uploadedvideo.jpg" frameborder="0" height="400"
            width="95%"></iframe></p>
        </div>

        <script>
        // function myFunction() {
        // var checkBox = document.getElementById("myCheck");
        // var text = document.getElementById("text");
        // var image=document.getElementById("image");
        // if (checkBox.checked == true){
        //   if (document.getElementById("uploadi")){
        //     text.style.display = "block";
        //   }else if (document.getElementById("uploadt")) {
        //     image.style.display = "block";
        //   }
        // } else {
        //     text.style.display = "none";
        // }
        // }
        </script>




        <!-- .......... -->

        <!-- <script>
        function myFunction() {
        var checkBox = document.getElementById("myCheck");
        var text = document.getElementById("text");
        if (checkBox.checked == true){
            text.style.display = "block";
        } else {
            text.style.display = "none";
        }
        }
        </script> -->



    </section>


<br><br><br><br>
<!--for the output box-->
    <section class="container outputtext text-center" id="section3">
        <h2>Abstractive Summary</h2>
        <br><br>
        <button onclick="myFunction()" type="submit" name="submit" class="upload " style="margin-left:20px">Obtain summary</button>
        <br><br><br>
        <input type="checkbox" id="myCheck1" onclick="myOutput()">
        <label for="myCheck1"><h3>Check the box to view the Summary.</h3></label>

        <!--<p id="text" style="display:none">Checkbox is CHECKED!</p>-->
        <div id="output" style="display:none">
            <p><iframe src="uploads/obtainedsummary.txt" frameborder="0" height="150"
            width="95%"></iframe></p>
        </div>
        <script>
            function myFunction() {
              <?php
                  $command = escapeshellcmd('python evaluation.py');
                  $output = shell_exec($command);
                  echo $output;
              ?>
            }
        </script>

        <script>
        // function myOutput() {
        // var checkBox1 = document.getElementById("myCheck1");
        // var text = document.getElementById("output");
        // if (checkBox1.checked == true){
        //     text.style.display = "block";
        // } else {
        //     text.style.display = "none";
        // }
        // }
        </script>
    </section>
    <br><br><br>


</body>
</html>
