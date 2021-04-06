# STA 141C Big-data and Statistical Computing

## Discussion 2: Working on a Remote Server

TA: Tesi Xiao


### Connecting to a Remote Server

The statisitcs department grant the class access to the department server. You can find the instructions [here](https://anson.ucdavis.edu/systems/). The campus login ID is requied. The connection between the local and the server is based on the SSH protocal. 

> The [SSH](https://www.ssh.com/ssh/) (Secure Shell) protocol uses encryption to secure the connection between a client and a server. All user authentication, commands, output, and file transfers are encrypted to protect against attacks in the network.

 
Note that if you are off-campus, you must connect to UC Davis [VPN](https://www.library.ucdavis.edu/service/connect-from-off-campus/) for the access to the department servers.


- MacOS/Linux Users

    Open "Terminal" and enter `ssh username@hostname.ucdavis.edu` and press `return`. Then you will be prompted to enter your password. You may need to reset your password for the first time.

- Windows Users

    1. Connect via the third party SSH client "PuTTY"
    
        - Download "putty" from http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html
        - Launch putty
        - Enter appropriate hostname in the "Host Name" box
        - Click open. You will be prompted for your loginID and pasword.

    2. Connect via bash
    
        Fortunately, Microsoft has built an integrated OpenSSH client to Windows in 2018. Check out this [post](https://www.howtogeek.com/336775/how-to-enable-and-use-windows-10s-built-in-ssh-commands/). Then you can connect to an Secure Shell server via the command `ssh` in bash from Windows without installing PuTTY or any other third-party software.
        

        
### Basic Linux Commands


The remote server usually are linux servers without the graphical user interface. You can only enter linux commands to manipulate files, execute scripts, etc. Below are several useful commands.


- `ls` - list files in the current directory
- `cd`	- change directory
- `pwd`	- show current directory
- `mkdir`	- create a directory
- `cp`	- copy a file
- `mv`	- move a file
- `rm`	- remove a file/s
- `cat`	- view text files at once
- `more`	- view text files one page at a time
- `nano`	- basic text editor
- `vim`	- more powerful text editor
- `[tab]`	- auto-completes file names and commands
- `[up][down]` 	- browse through previously run commands
- `submit`	- allows you to submit a job to run in the background. You will be notified via  email when your job has completed.
- `nps`	- shows jobs currently running on the server
- `kill`	- allows you to terminate a job you own. specify the process id. (e.g. kill 1234)
- `exit`	- logout


### SSH Secure File Transfer

There are certain occasions when you need to transfer files from your local system to the remote server or from the server to the local. For example, you can upload the local python script to the server and run your script remotely. Also, you may need to download the output of your job from the server and create plots on your local machine. 

> [SFTP](https://www.ssh.com/ssh/sftp/) (SSH File Transfer Protocol) is a secure file transfer protocol. It runs over the SSH protocol. It supports the full security and authentication functionality of SSH.


- Transfer files from local to remote

    Suppose you have your python script `myfile.py` in your local directory and you want to upload it to the server.
    
    1. Make sure you are in the terminal (bash) of your local machine.
    2. Enter the following command: `scp directoryOfYourFile username@hostname.ucdavis.edu:~/foldername/`
        
- Submit jobs:

    Now you have `myfile.py` on the server. Then you can submit this job to run in the background. You will be notified via  email when your job has completed.
    
    1. log in to the server. Use command ‘cd foldername’ to change directory and check whether your files are there.
    2. submit jobs by the following command: `submit myfile.py`.
    
- Transfer files from remote to local:

    When your job finish, your result will be saved (your can change the command in your `.py` file to change the save location) and the printout output (error and warnings also) in a file with `.out`. 
    
    In the local terminal, enter `scp username@alan.ucdavis.edu:~/foldername/filename.out MyTargetLocation` to download the file `filename.out` to `MyTargetLocation`.
