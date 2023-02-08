<?php
if(isset($_POST['submit'])){
    $file = $_FILES['file'];


    $fileName = $_FILES['file']['name'];
    $fileTmpName= $_FILES['file']['tmp_name'];
    $fileSize = $_FILES['file']['size'];
    $fileError = $_FILES['file']['error'];
    $fileType = $_FILES['file']['type'];

    $fileExt = explode('.', $fileName);
    $fileActualExt = strtolower(end($fileExt));

    $allowed = array('mp4', 'm4v', 'mpg','avi');


    if(in_array($fileActualExt, $allowed))
    {
        if($fileError === 0){
            if($fileSize < 8000000)
            {
                $fileNameNew = 'uploadedvideo'.'.'.$fileActualExt;
                $fileDestination = 'uploads/'.$fileNameNew;

                $moved = move_uploaded_file($fileTmpName, $fileDestination);

                header("Location:video.php?uploadsuccess");

            }else{
                echo "The file is too big!";
            }

        }
        else{
            echo "There was an error uploading your file!";
        }
    }
    else
    {
        echo "Files with extension mp4,m4v,mpg,avi only are allowed";
    }

}
?>
