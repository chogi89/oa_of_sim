#include "ros/ros.h"
#include "oa_of/MsgOAOF.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <iostream>
#include <ctype.h>
#include <math.h>

#define S_TIME  0.01

#define	WIDTH	80
#define HEIGHT	30

#define D_SET   0.5
#define R_CHECK 0.1

using namespace cv;
using namespace std;

double pose_p_x_c = 0;
double pose_p_y_c = 0;
double pose_p_z_c = 0;
double pose_o_qx_c = 0;
double pose_o_qy_c = 0;
double pose_o_qz_c = 0;
double pose_o_qw_c = 1;
double pose_o_ex_c = 0;
double pose_o_ey_c = 0;
double pose_o_ez_c = 0;

double pose_p_x_t = 0;
double pose_p_y_t = 0;
double pose_p_z_t = 1.7;
double pose_o_qx_t = 0;
double pose_o_qy_t = 0;
double pose_o_qz_t = 0;
double pose_o_qw_t = 1;
double pose_o_ex_t = 0;
double pose_o_ey_t = 0;
double pose_o_ez_t = 0;

double cy = 0;
double sy = 0;
double cr = 0;
double sr = 0;
double cp = 0;
double sp = 0;

double d_c2t = 0;

void msgCallback(const geometry_msgs::PoseStamped::ConstPtr& Pose){
	pose_p_x_c = Pose->pose.position.x;
	pose_p_y_c = Pose->pose.position.y;
	pose_p_z_c = Pose->pose.position.z;
	pose_o_qx_c = Pose->pose.orientation.x;
	pose_o_qy_c = Pose->pose.orientation.y;
	pose_o_qz_c = Pose->pose.orientation.z;
	pose_o_qw_c = Pose->pose.orientation.w;
}

int main (int argc, char **argv){
	ros::init(argc, argv, "oa_of_hovering");
	ros::NodeHandle nh, nh_mavros;

	ros::Publisher oa_of_pub = nh.advertise<oa_of::MsgOAOF>("oa_of_msg",100);

    ros::Publisher Set_Position_pub = nh_mavros.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local",100);
    ros::Subscriber oa_of_sub = nh_mavros.subscribe("mavros/local_position/pose", 10, msgCallback);

    ros::Rate loop_rate(1/S_TIME);

    Mat mat_rgb, mat_gray;
    Mat mat_arrow(Size(WIDTH*100, HEIGHT*100),CV_8UC1,255);

    VideoCapture cap;

	char keypressed;

    uchar arr_gray_prev[WIDTH][HEIGHT];
	uchar arr_gray_curr[WIDTH][HEIGHT];

	double Ix[WIDTH-1][HEIGHT-1];
	double Iy[WIDTH-1][HEIGHT-1];
	double It[WIDTH-1][HEIGHT-1];

	double u[WIDTH-1][HEIGHT-1];
	double v[WIDTH-1][HEIGHT-1];

	double mu_u[WIDTH-1][HEIGHT-1];
	double mu_v[WIDTH-1][HEIGHT-1];

    CvPoint p1[WIDTH-1][HEIGHT-1];
    CvPoint p2[WIDTH-1][HEIGHT-1];

	double alpha=1;

	double u_rs[WIDTH*HEIGHT];
	double v_rs[WIDTH*HEIGHT];

	double OFright = 0;
	double OFleft = 0;
	double Psi_of = 0;

	cap.open(-1);
	if(!cap.isOpened()){
		cout << "***Could not initialize capturing...***\n";
		return -1;
	}

    for(int i=0; i<(WIDTH-1); i++){
		for(int j=0; j<(HEIGHT-1); j++){
            p1[i][j] = cvPoint(i,j);
		}
    }

    for(int i=0; i<(WIDTH-1); i++){
	    for(int j=0; j<(HEIGHT-1); j++){
            p1[i][j].x = (p1[i][j].x*100)+100;
            p1[i][j].y = (p1[i][j].y*100)+100;
        }
    }

    double count = 0;
	while (ros::ok()){
        mat_arrow.setTo(255);
		cap >> mat_rgb;
		if(mat_rgb.empty()){
			break;
        }
		resize(mat_rgb, mat_rgb, Size(WIDTH, HEIGHT));
        cvtColor(mat_rgb, mat_gray, CV_BGR2GRAY);

		for(int i=0; i < (WIDTH); i++){
			for(int j=0; j < (HEIGHT); j++){
				arr_gray_prev[i][j] = arr_gray_curr[i][j];
			}
		}
		for(int i=0; i < (WIDTH); i++){
			for(int j=0; j < (HEIGHT); j++){
        			arr_gray_curr[i][j] = mat_gray.at<uchar>(j,i);
			}
		}

		// ------------------------------ //
		// -- Optical Flow Calculation -- //
        // ------------------------------ //

		for(int i=0; i<(WIDTH-1); i++){
			for(int j=0; j<(HEIGHT-1); j++){
				Ix[i][j] = (arr_gray_prev[i+1][j] - arr_gray_prev[i][j] + arr_gray_prev[i+1][j+1] - arr_gray_prev[i][j+1] + arr_gray_curr[i+1][j] - arr_gray_curr[i][j] + arr_gray_curr[i+1][j+1] - arr_gray_curr[i][j+1])/4;
				Iy[i][j] = (arr_gray_prev[i][j+1] - arr_gray_prev[i][j] + arr_gray_prev[i+1][j+1] - arr_gray_prev[i+1][j] + arr_gray_curr[i][j+1] - arr_gray_curr[i][j] + arr_gray_curr[i+1][j+1] - arr_gray_curr[i+1][j])/4;
				It[i][j] = (arr_gray_curr[i][j] - arr_gray_prev[i][j] + arr_gray_curr[i+1][j] - arr_gray_prev[i+1][j] + arr_gray_curr[i][j+1] - arr_gray_prev[i][j+1] + arr_gray_curr[i+1][j+1] - arr_gray_prev[i+1][j+1])/4;
			}
		}

		for(int i=0; i<(WIDTH-2); i++){
		    for(int j=0; j<(HEIGHT-2); j++){
                mu_u[i][j] = (u[i][j+1] + u[i+1][j] + u[i+2][j+1] + u[i+1][j+2])/6 + (u[i][j] + u[i][j+2] + u[i+2][j] + u[i+2][j+2])/12;
                mu_v[i][j] = (v[i][j+1] + v[i+1][j] + v[i+2][j+1] + v[i+1][j+2])/6 + (v[i][j] + v[i][j+2] + v[i+2][j] + v[i+2][j+2])/12;
		    }
		}

		for(int i=0; i<(WIDTH-2); i++){
		    for(int j=0; j<(HEIGHT-2); j++){
                u[i][j] = mu_u[i][j] - Ix[i][j]*((Ix[i][j]*mu_u[i][j] + Iy[i][j]*mu_v[i][j] + It[i][j])/(alpha*alpha + Ix[i][j]*Ix[i][j] + Iy[i][j]*Iy[i][j]));
                v[i][j] = mu_v[i][j] - Iy[i][j]*((Ix[i][j]*mu_u[i][j] + Iy[i][j]*mu_v[i][j] + It[i][j])/(alpha*alpha + Ix[i][j]*Ix[i][j] + Iy[i][j]*Iy[i][j]));
		    }
		}

		for(int i=0; i<(WIDTH-1); i++){
		    for(int j=0; j<(HEIGHT-1); j++){
                p2[i][j].x = p1[i][j].x+(int)(u[i][j]*2);
                p2[i][j].y = p1[i][j].y+(int)(v[i][j]*2);
		    }
		}

        for(int i=0; i<(WIDTH-1); i++){
		    for(int j=0; j<(HEIGHT-1); j++){
                arrowedLine(mat_arrow,p1[i][j],p2[i][j],0,3,CV_AA,0,1);
		    }
		}

        // --------------------- //
        // -- FOE calculation -- //
        // --------------------- //

        for(int i=0; i<(WIDTH-1); i++){
            for(int j=0; j<(HEIGHT-1); j++){
                u_rs[i*HEIGHT+j] = u[i][j];
                v_rs[i*HEIGHT+j] = v[i][j];
            }
        }
        //u_rs = reshape(u);
        //v_rs = reshape(v);
/*
        A = transpose([u_rs; v_rs]);

        for i = 1 : ((width-1)*(height-1))
            b(i) = x_rs(i)*v_rs(i) - y_rs(i)*u_rs(i);
        end
        b = transpose(b);

        if det(transpose(A)*A) ~= 0
            FOE = inv(transpose(A)*A)*transpose(A)*b;

            if FOE(1) > width/2-2
                FOE(1) = width/2-2;
            end
            if FOE(1) < -1*(width/2-2)
                FOE(1) = -1*(width/2-2);
            end

            FOE = round(FOE);
        end
*/
        // ----------------------------------------- //
		// -- Computation Horizontal Optical Flow -- //
        // ----------------------------------------- //

        OFright = 0;
        OFleft = 0;

        for (int i=0; i<((WIDTH/2)-2); i++){            // Size of u,v are (WIDTH-1) and (HEIGHT-1), respectively.
            for(int j=0; j<(HEIGHT-2); j++){
                OFright = OFright + sqrt((u[i][j]*u[i][j]) + (v[i][j]*v[i][j]));
            }
        }
        for (int i=((WIDTH/2)); i<(WIDTH-2); i++){      // Size of u,v are (WIDTH-1) and (HEIGHT-1), respectively.
            for(int j=0; j<(HEIGHT-2); j++){
                OFleft = OFleft + sqrt((u[i][j]*u[i][j]) + (v[i][j]*v[i][j]));
            }
        }

        // ------------------------------ //
        // -- Heading angle Generation -- //
        // ------------------------------ //

        Psi_of  = -0.05*(OFright - OFleft);

        // -------------------------------- //
        // -- Target Position Generation -- //
        // -------------------------------- //

        if(count>=5){
            d_c2t = sqrt((pose_p_x_c-pose_p_x_t)*(pose_p_x_c-pose_p_x_t) + (pose_p_y_c-pose_p_y_t)*(pose_p_y_c-pose_p_y_t));

            if(d_c2t < R_CHECK){
                pose_o_ex_t = 0;
                pose_o_ey_t = 0;
                pose_o_ez_t = 0;
                pose_p_x_t = 0;
                pose_p_y_t = 0;
                pose_p_z_t = 1.7;

                cy = cos(pose_o_ez_t * 0.5);
                sy = sin(pose_o_ez_t * 0.5);
                cr = cos(pose_o_ey_t * 0.5);
                sr = sin(pose_o_ey_t * 0.5);
                cp = cos(pose_o_ex_t * 0.5);
                sp = sin(pose_o_ex_t * 0.5);

                pose_o_qw_t = cy * cr * cp + sy * sr * sp;
                pose_o_qx_t = cy * sr * cp - sy * cr * sp;
                pose_o_qy_t = cy * cr * sp + sy * sr * cp;
                pose_o_qz_t = sy * cr * cp - cy * sr * sp;
            }
        }

        // ------------- //
        // -- Display -- //
        // ------------- //

        namedWindow("Camera",WINDOW_NORMAL);
		imshow("Camera", mat_gray);
		//namedWindow("Optical_flow",WINDOW_NORMAL);
		//imshow("Optical_flow",mat_arrow);

		keypressed = (char)waitKey(10);
		if(keypressed == 27)
			break;

        oa_of::MsgOAOF msg;
        geometry_msgs::PoseStamped msg_setposition;

		msg.data = count;
		msg_setposition.pose.position.x = pose_p_x_t;
		msg_setposition.pose.position.y = pose_p_y_t;
		msg_setposition.pose.position.z = pose_p_z_t;
		msg_setposition.pose.orientation.x = pose_o_qx_t;
		msg_setposition.pose.orientation.y = pose_o_qy_t;
		msg_setposition.pose.orientation.z = pose_o_qz_t;
		msg_setposition.pose.orientation.w = pose_o_qw_t;

		oa_of_pub.publish(msg);
		Set_Position_pub.publish(msg_setposition);

		ROS_INFO("Send msg = %f", count);
		ROS_INFO("Target X = %f", pose_p_x_t);
		ROS_INFO("Target Y = %f", pose_p_y_t);
		ROS_INFO("Target Yaw = %f", pose_o_ez_t);

        ros::spinOnce();
        //r.sleep();

        loop_rate.sleep();
		count = count + S_TIME;
	}
    cap.release();

	return 0;
}
