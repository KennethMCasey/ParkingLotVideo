using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Emgu;
using System.Threading;
using Dapper;
using System.Data;
using System.Data.SqlClient;


namespace Carz
{
    class VideoInterpreter
    {
        private bool showWindow;
        private int numCarsInLot;

        private string parkingLotName;
        private int parkingLotId;
        private string pathToVideoSource;

        private bool didStart;

        private Action<DateTime, VideoInterpreter> CarDidEnter;
        private Action<DateTime, VideoInterpreter> CarDidLeave;

        private double fps;
        private double frameWidth;
        private double frameHeight;

        private Emgu.CV.VideoCapture vCapture;
        private Emgu.CV.CascadeClassifier casc;

        public VideoInterpreter(string pathToVideoSource, string pathToXmlFile, int parkingLotId, string parkingLotName)
        {
            //sets variables that are not passed
            this.showWindow = false;
            numCarsInLot = 0;
            fps = 30;
            frameWidth = 1920;
            frameHeight = 1080;
            didStart = false;

            //sets parking lot information
            this.parkingLotName = parkingLotName;
            this.parkingLotId = parkingLotId;
            this.pathToVideoSource = pathToVideoSource;

            //sets up video stream pull
            vCapture = new Emgu.CV.VideoCapture(pathToVideoSource);
            vCapture.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps, fps);
            vCapture.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth, frameWidth);
            vCapture.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight, frameHeight);
            casc = new Emgu.CV.CascadeClassifier(pathToXmlFile);
        }

        public async void start()
        {

            if (this.CarDidEnter == null) CarDidEnter = CarDidEnterDefault;
            if (this.CarDidLeave == null) CarDidLeave = CarDidLeaveDefault;

            //flag to note process started to prevent changing peramaters that would break the processing
            didStart = true;

            //sets up the image matrixs for the input raw and output processed
            Emgu.CV.Mat iMatrix = new Emgu.CV.Mat();
            Emgu.CV.Mat oMatrix = new Emgu.CV.Mat();

            await Task.Run(() =>
            {
                //creates a window if desired, window for testing purposes 
                if (showWindow) Emgu.CV.CvInvoke.NamedWindow("Car Detection Test", Emgu.CV.CvEnum.NamedWindowType.FreeRatio);


                //This async task continually pulls the video frame to be processed.
                Task.Run(() =>
                {
                    for (; ; )
                    {
                        vCapture.Read(iMatrix);
                        //System.Console.Out.WriteLine(iMatrix.Size.ToString());
                        Emgu.CV.CvInvoke.WaitKey((int)(1 / fps * 1000));
                    }
                });

                //This async task continually process the pulled frames
                for (; ; )
                {

                    //If the matrix used to store the frames is not empty
                    if (!iMatrix.IsEmpty)
                    {
                        //Converts the contents of the imatrix to greyscale, export results to omatrix
                        //this is to ensure proper processing
                        Emgu.CV.CvInvoke.CvtColor(iMatrix, oMatrix, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                        //Uses the cascade xml file provided in the initalizer to draw rectangles arround possible canditates.
                        Rectangle[] rects = casc.DetectMultiScale(oMatrix, 1.025, 37, new Size(250, 250), new Size(600, 600));

                        //removes the image from the out matrix if one exists to make room for the new one.
                        oMatrix.PopBack(1);

                        //sets inital value to zero
                        int carsInFrame = 0;

                        //loops through all of the rectangles in the discorvered object array
                        foreach (Rectangle rect in rects)
                        {
                            //draws the rectangles on the imatrix image for display if we wish to show the window
                            //we use imatrix as it is in color
                            if (showWindow)
                            {
                                var x = rect.X;
                                var y = rect.Y;
                                var w = rect.Width;
                                var h = rect.Height;
                                Emgu.CV.CvInvoke.Rectangle(iMatrix, new Rectangle(x, y, w, h), new Emgu.CV.Structure.MCvScalar(50));
                            }
                            //increase the number of cars in frame for each iteration
                            carsInFrame++;
                        }
                        System.Console.Out.Write(numCarsInLot);



                        //if the number of cars has changed
                        //call the proper delagte the necessary amount of times
                        if (carsInFrame > numCarsInLot) for (int i = 0; i < carsInFrame - numCarsInLot; i++) Task.Run(() => CarDidEnter.Invoke(DateTime.Now, this));
                        if (carsInFrame < numCarsInLot) for (int i = 0; i < numCarsInLot - carsInFrame; i++) Task.Run(() => CarDidLeave.Invoke(DateTime.Now, this));

                        //update the number of cars
                        numCarsInLot = carsInFrame;


                        //if the show window flag is true we push the drawn images to the window
                        if (showWindow) Emgu.CV.CvInvoke.Imshow("Car Detection Test", iMatrix);

                        //discard the now rendered frame
                        iMatrix.PopBack(1);

                        //Distroys windows and stops loop if the escape key is pressed
                        if (showWindow && Emgu.CV.CvInvoke.WaitKey(33) == 27)
                        {
                            if (showWindow) Emgu.CV.CvInvoke.DestroyAllWindows();
                            break;
                        }
                    }
                }
            });
        }

        //This will be called every time a car leaves async
        public void setCarDidLeaveDelegate(Action<DateTime, VideoInterpreter> func) { if (!didStart) CarDidLeave = func; }

        //This will be called every time a car enteres async
        public void setCarDidEnterDelegate(Action<DateTime, VideoInterpreter> func) { if (!didStart) CarDidEnter = func; }

        //set this flag for testing, will make a window to see the results. default false
        public void setShowWindow(bool flag) { if (!didStart) this.showWindow = flag; }

        //overides the 30fps count if needed
        public void setfps(double fps) { if (!didStart) this.fps = fps; }

        //overides the 1920 frame width if needed
        public void setFrameWidth(double width) { if (!didStart) this.frameWidth = width; }

        //overides the 1080 frame height if needed
        public void setFrameHeight(double height) { if (!didStart) this.frameHeight = height; }

        //some getters that can be useful in delegate functions
        public string getPathToVideoSource() { return pathToVideoSource; }
        public int getParkingLotId() { return parkingLotId; }
        public string getParkingLotName() { return parkingLotName; }

        /*
         Examples of delegate usage...

         public void CarDidLeave(DateTime timeCarLeft, VideoInterpreter sender) 
         {
         timeCarLeft is the time this was called
         use the sender object to get attributes such as the lot id and lot name
         }

         public void CarDidEnter(DateTime timeCarEntered, VideoInterpreter sender) 
         {
         timeCarLeft is the time this was called
         use the sender object to get attributes such as the lot id and lot name
         } 
         
         */
        private void CarDidEnterDefault(DateTime timeCarLeft, VideoInterpreter sender)
        {
            using (SqlConnection connection = new SqlConnection(SQLConnection.ConnString("ParkingLotDB")))
            {
                try
                {
                    using (SqlCommand command = new SqlCommand("spCarParked", connection))
                    {
                        command.CommandType = CommandType.StoredProcedure;
                        command.Parameters.Add(new SqlParameter("@ParkingLotID", sender.getParkingLotId()));
                        connection.Open();
                        command.ExecuteNonQuery();
                        connection.Close();
                        System.Console.Out.Write("Success");
                    }
                }
                catch (Exception e)
                {
                    System.Console.Out.Write(e.ToString());
                }
            }
        }


        private void CarDidLeaveDefault(DateTime timeCarLeft, VideoInterpreter sender)
        {
            using (SqlConnection connection = new SqlConnection(SQLConnection.ConnString("ParkingLotDB")))
            {

                try
                {
                    using (SqlCommand command = new SqlCommand("spCarLeft", connection))
                    {
                        command.CommandType = CommandType.StoredProcedure;
                        command.Parameters.Add(new SqlParameter("@ParkingLotID", sender.getParkingLotId()));
                        connection.Open();
                        command.ExecuteNonQuery();
                        connection.Close();
                        System.Console.Out.Write("Success");
                    }
                }
                catch (Exception e)
                {
                    System.Console.Out.Write(e.ToString());
                }
            }
        }
    }
}
