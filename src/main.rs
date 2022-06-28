mod motion_detection_camera {
    use once_cell::sync::Lazy;

    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread;

    use chrono::{DateTime, Duration, Local};
    use opencv::core::{absdiff, convert_scale_abs, no_array, Point, Scalar, Size_, CV_32F};
    use opencv::imgproc::{
        draw_contours, CHAIN_APPROX_SIMPLE, LINE_8, RETR_EXTERNAL, THRESH_BINARY,
    };
    use opencv::prelude::*;
    use opencv::types::VectorOfMat;
    use opencv::videoio::{
        VideoCapture, VideoWriter, CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_FRAME_WIDTH,
    };
    use opencv::{highgui, imgproc, Result};

    pub enum RecorderRequest {
        Start,
        Frame(Mat),
        Stop,
        Shutdown,
    }

    #[derive(Debug, PartialEq, Eq)]
    pub enum RecorderResponse {
        Ok,
        Err,
    }

    #[derive(Debug, Clone)]
    pub struct RecorderOption {
        fourcc: i32,
        fps: f64,
        frame_size: Size_<i32>,
        is_color: bool,
    }

    impl RecorderOption {
        pub fn new(fourcc: i32, fps: f64, frame_size: Size_<i32>, is_color: bool) -> Self {
            Self {
                fourcc,
                fps,
                frame_size,
                is_color,
            }
        }

        pub fn new_with_cap(fourcc: i32) -> (Self, VideoCapture) {
            let cap = VideoCapture::new(0, CAP_ANY).unwrap();

            let opened = VideoCapture::is_opened(&cap).unwrap();
            if !opened {
                panic!("Unable to open default camera!");
            }

            let height = cap.get(CAP_PROP_FRAME_HEIGHT).unwrap();
            let width = cap.get(CAP_PROP_FRAME_WIDTH).unwrap();
            let fps = cap.get(CAP_PROP_FPS).unwrap();

            (
                Self::new(
                    fourcc,
                    fps,
                    Size_ {
                        width: width as i32,
                        height: height as i32,
                    },
                    true,
                ),
                cap,
            )
        }
    }

    fn recorder_thread(
        rec_option: RecorderOption,
        req_receiver: Receiver<RecorderRequest>,
        res_sender: Sender<RecorderResponse>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut recording = false;
            let mut writer = None;

            while let Ok(req) = req_receiver.recv() {
                match req {
                    RecorderRequest::Start => {
                        writer = Some(Box::new(
                            VideoWriter::new(
                                &format!("{}.mp4", uuid::Uuid::new_v4()),
                                rec_option.fourcc,
                                rec_option.fps,
                                rec_option.frame_size,
                                rec_option.is_color,
                            )
                            .unwrap(),
                        ));

                        recording = true;
                        res_sender.send(RecorderResponse::Ok).unwrap();
                    }
                    RecorderRequest::Frame(mat) => {
                        if recording {
                            if let Some(writer) = writer.as_mut() {
                                writer.write(&mat).unwrap();

                                res_sender.send(RecorderResponse::Ok).unwrap();
                            } else {
                                res_sender.send(RecorderResponse::Err).unwrap();
                            }
                        } else {
                            res_sender.send(RecorderResponse::Err).unwrap();
                        }
                    }
                    RecorderRequest::Stop => {
                        if recording {
                            if let Some(writer) = writer.as_mut() {
                                writer.release().unwrap();

                                res_sender.send(RecorderResponse::Ok).unwrap();
                            } else {
                                res_sender.send(RecorderResponse::Err).unwrap();
                            }

                            writer = None;
                            recording = false;
                        } else {
                            res_sender.send(RecorderResponse::Err).unwrap();
                        }
                    }
                    RecorderRequest::Shutdown => {
                        res_sender.send(RecorderResponse::Ok).unwrap();
                        break;
                    }
                }
            }
        })
    }

    pub struct RecorderClient {
        req_sender: Sender<RecorderRequest>,
        res_receiver: Receiver<RecorderResponse>,
    }

    impl RecorderClient {
        pub fn new(
            req_sender: Sender<RecorderRequest>,
            res_receiver: Receiver<RecorderResponse>,
        ) -> Self {
            Self {
                req_sender,
                res_receiver,
            }
        }

        pub fn send_request(&self, request: RecorderRequest) -> RecorderResponse {
            self.req_sender.send(request).unwrap();

            self.res_receiver.recv().unwrap()
        }
    }

    pub fn start_recorder_thread(
        rec_option: &RecorderOption,
    ) -> (RecorderClient, thread::JoinHandle<()>) {
        let (req_sender, req_receiver) = mpsc::channel();
        let (res_sender, res_receiver) = mpsc::channel();

        let rec_thread = recorder_thread(rec_option.clone(), req_receiver, res_sender);

        (RecorderClient::new(req_sender, res_receiver), rec_thread)
    }

    #[allow(dead_code)]
    pub enum ColorMode {
        Normal,
        Gray,
        FrameDelta,
    }

    pub struct MovingDetectCameraOption {
        pub color_mode: ColorMode,
        pub plot_contours: bool,
    }

    static DEFAULT_RECORDING_MIN_LEN: Lazy<Duration> =
        Lazy::new(|| Duration::from_std(std::time::Duration::from_secs(3)).unwrap());

    pub fn start_moving_detection_camera(
        mut cap: VideoCapture,
        rec_client: &RecorderClient,
        mdc_option: MovingDetectCameraOption,
    ) -> Result<()> {
        init();

        let window_name = "VideoCapture";
        highgui::named_window(window_name, 1)?;

        let mut have_avg = false;
        let mut avg = Mat::default();

        let mut is_recording = false;
        let mut start_datetime: Option<DateTime<Local>> = None;

        loop {
            let mut frame = Mat::default();
            cap.read(&mut frame)?;

            if frame.size()?.width > 0 {
                let mut gray = Mat::default();
                imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

                if !have_avg {
                    gray.convert_to(&mut avg, CV_32F, 1., 0.)?;

                    have_avg = true;
                }

                imgproc::accumulate_weighted(&gray, &mut avg, 0.6, &no_array())?;
                let mut scale_abs = Mat::default();
                convert_scale_abs(&avg, &mut scale_abs, 1., 0.)?;
                let mut frame_delta = Mat::default();
                absdiff(&gray, &scale_abs, &mut frame_delta)?;

                // 平均画素との差分...
                let threshold1 = 40.;

                let mut thresh = Mat::default();
                imgproc::threshold(&frame_delta, &mut thresh, threshold1, 255., THRESH_BINARY)?;

                let mut contours = VectorOfMat::default();

                let (trans_to_bgr_flag, mut frame) = match mdc_option.color_mode {
                    ColorMode::Normal => (false, frame),          // もとの画像
                    ColorMode::Gray => (true, gray),              // グレースケール
                    ColorMode::FrameDelta => (true, frame_delta), // 動きのあった画素
                };
                if trans_to_bgr_flag {
                    imgproc::cvt_color(&frame.clone(), &mut frame, imgproc::COLOR_GRAY2BGR, 0)?;
                }

                imgproc::find_contours(
                    &thresh,
                    &mut contours,
                    RETR_EXTERNAL,
                    CHAIN_APPROX_SIMPLE,
                    Point::default(),
                )?;

                if mdc_option.plot_contours {
                    draw_contours(
                        &mut frame,
                        &contours,
                        -1,
                        Scalar::new(0., 0., 255., 0.),
                        3,
                        LINE_8,
                        &no_array(),
                        i32::MAX,
                        Point::default(),
                    )?;
                }

                highgui::imshow(window_name, &frame)?;

                // recoding function
                {
                    println!("contours: {:?}", contours.len());
                    println!(
                        " - is_recording: {is_recording:?}, start_datetime: {start_datetime:?}"
                    );
                    let contours_threshold = 50;
                    if contours.len() >= contours_threshold {
                        // しきい値以上に、平均画素から異なる画素が検出された場合、録画を開始する
                        let now = chrono::Local::now();
                        println!("move detect! at {:?}", now);
                        println!("contours: {:?}", contours.len());

                        if !is_recording {
                            is_recording = true;

                            if let RecorderResponse::Err =
                                rec_client.send_request(RecorderRequest::Start)
                            {
                                panic!("staring record failed!");
                            }
                        }
                        start_datetime = Some(now);
                    } else if is_recording {
                        // 録画条件を確認し、満たさなければ停止
                        let now = chrono::Local::now();
                        if let Some(start_datetime) = start_datetime {
                            let diff = now - start_datetime;
                            println!("recording stop.... ? {:?}", diff);
                            if diff >= *DEFAULT_RECORDING_MIN_LEN {
                                println!(" - recording STOP!!!!!!!!!");

                                if let RecorderResponse::Err =
                                    rec_client.send_request(RecorderRequest::Stop)
                                {
                                    panic!("stopping record failed!");
                                }

                                is_recording = false;
                            }
                        }
                    }

                    if is_recording {
                        // フレーム追加
                        println!("recording......");
                        if let RecorderResponse::Err =
                            rec_client.send_request(RecorderRequest::Frame(frame.clone()))
                        {
                            panic!("recording failed!");
                        }
                    }
                }

                if highgui::wait_key(10)? > 0 {
                    break;
                }
            }
        }

        Ok(())
    }

    static mut INITIALIZED: bool = false;

    pub fn init() {
        let initialized = unsafe { INITIALIZED };

        if !initialized {
            unsafe {
                INITIALIZED = true;
            }

            env_logger::init();
        }
    }
}

fn main() {
    use crate::motion_detection_camera::{
        start_moving_detection_camera, start_recorder_thread, ColorMode, MovingDetectCameraOption,
        RecorderOption, RecorderRequest, RecorderResponse,
    };
    use opencv::videoio::VideoWriter;

    let (rec_option, cap) = RecorderOption::new_with_cap(
        VideoWriter::fourcc('m' as i8, 'p' as i8, '4' as i8, 'v' as i8).unwrap(),
    );

    let (rec_client, thread) = start_recorder_thread(&rec_option);

    let mdc_option = MovingDetectCameraOption {
        color_mode: ColorMode::Normal,
        plot_contours: true,
    };

    start_moving_detection_camera(cap, &rec_client, mdc_option).unwrap();

    assert!(RecorderResponse::Ok == rec_client.send_request(RecorderRequest::Shutdown));

    thread.join().unwrap();
}
