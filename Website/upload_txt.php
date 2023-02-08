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

    $allowed = array('txt', 'odt', 'docx');


    if(in_array($fileActualExt, $allowed))
    {
        if($fileError === 0){
            if($fileSize < 1000000)
            {
                $fileNameNew = 'uploadedtext'.'.'.$fileActualExt;
                $fileDestination = 'uploads/'.$fileNameNew;

                $moved = move_uploaded_file($fileTmpName, $fileDestination);

                header("Location:text.php?uploadsuccess");

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
        echo "Files with extension txt, odt, docx only are allowed";
    }

}
?>
