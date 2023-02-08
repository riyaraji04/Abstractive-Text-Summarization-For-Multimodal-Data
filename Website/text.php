
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
        <li  class="active"><a href="about.php">About Us</a></li>

      </ul>
      <!-- <ul class="nav navbar-nav navbar-right">
        <li><a href="#"><span class="glyphicon glyphicon-user"></span> Sign Up</a></li>
        <li><a href="#"><span class="glyphicon glyphicon-log-in"></span> Login</a></li>
      </ul> -->
    </div>
  </nav>

    <br><br><br><br>

    <div class="container text-center" id="section1">
      <h3>Upload a text file below</h3>
        <form id="uploadt" action="upload_txt.php" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" class="browse" style="margin-left: 400px;"><br>
            <button type="submit" name="submit" class="upload "style="margin-left:40px" >Upload Text</button>
        </form>
        <br><br><br>
        <button type="button" name="button" class="upload" onclick="document.location='index.php#section2'" style="margin-left:30px;">Click to view the Extracted Text</a></button>
    </div>




</body>
</html>
