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

    $allowed = array('jpg', 'jpeg', 'png');


    if(in_array($fileActualExt, $allowed))
    {
        if($fileError === 0){
            if($fileSize < 1000000)
            {
                //$fileNameNew = uniqid('',true).'.'.$fileActualExt;
                $fileNameNew = 'uploadedimage'.'.'.$fileActualExt;
                $fileDestination = 'uploads/'.$fileNameNew;

                $moved = move_uploaded_file($fileTmpName, $fileDestination);

                // ..........
                //
                // $command = escapeshellcmd('python convtext.py');
                // $output = shell_exec($command);
                // echo $output;

                // ..........


                header("Location:image.php?uploadsuccess");




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
        echo "Files with extension jpeg,jpg,png only are allowed";
    }

}
?>
