from win10toast import ToastNotifier
import time

def setnotification() :
    toaster = ToastNotifier()
    toaster.show_toast("It has been 20 min since you have looked away from your computer!!!",
                   "Please blink your eyes for sometime !!",
                    icon_path="eye.ico",
                    duration=10)

# Wait for threaded notification to finish
    while toaster.notification_active(): time.sleep(0.1)
if __name__ == '__main__':
    setnotification()
    #print ("main")