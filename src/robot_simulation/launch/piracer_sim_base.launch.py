"""Launch the baseline PiRacer model in Gazebo Classic."""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
import xacro


def _build_simulation(context):
    description_share = get_package_share_directory("robot_description")
    simulation_share = get_package_share_directory("robot_simulation")
    assets_share = get_package_share_directory("simulation_assets")
    gazebo_share = get_package_share_directory("gazebo_ros")

    xacro_path = os.path.join(description_share, "urdf", "piracer.urdf.xacro")
    default_world_path = os.path.join(
        simulation_share, "worlds", "sala_laboratoryjna_production.world"
    )
    world_path = LaunchConfiguration("world").perform(context) or default_world_path
    camera_enabled = LaunchConfiguration("camera_enabled").perform(context).lower()
    camera_update_rate = LaunchConfiguration("camera_update_rate").perform(context)
    yolo_enabled = LaunchConfiguration("yolo_enabled").perform(context).lower()
    yolo_backend = LaunchConfiguration("yolo_backend").perform(context).strip().lower()
    robot_description = xacro.process_file(
        xacro_path,
        mappings={
            "include_camera": camera_enabled,
            "camera_update_rate": camera_update_rate,
        },
    ).toxml()

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, "launch", "gzserver.launch.py")
        ),
        launch_arguments={
            "world": world_path,
            "factory": "true",
            "pause": LaunchConfiguration("start_paused"),
        }.items(),
    )
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, "launch", "gzclient.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("gui")),
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )
    ackermann_command_adapter = Node(
        package="piracer_gazebo_plugins",
        executable="ackermann_twist_adapter",
        namespace="piracer",
        parameters=[{"wheelbase_m": 0.155}],
        remappings=[("/piracer/cmd_vel", "/cmd_vel")],
        output="screen",
    )
    spawn_model = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description", "-entity", "piracer", "-timeout", "120",
            "-x", LaunchConfiguration("robot_spawn_x"),
            "-y", LaunchConfiguration("robot_spawn_y"),
            "-z", LaunchConfiguration("robot_spawn_z"),
            "-Y", LaunchConfiguration("robot_spawn_yaw"),
        ],
        output="screen",
    )
    suitcase_control_gui = Node(
        package="rqt_robot_steering",
        executable="rqt_robot_steering",
        condition=IfCondition(LaunchConfiguration("suitcase_control_gui")),
        output="screen",
    )
    camera_viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        arguments=[LaunchConfiguration("camera_view_topic")],
        condition=IfCondition(LaunchConfiguration("camera_viewer_enabled")),
        output="screen",
    )
    visual_odometry = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                simulation_share,
                "launch",
                "rgbd_visual_odometry.launch.py",
            )
        ),
        condition=IfCondition(
            LaunchConfiguration("visual_odometry_enabled")
            if camera_enabled in {"1", "true", "yes"}
            else "false"
        ),
    )
    yolo_nodes = []
    if yolo_enabled in {"1", "true", "yes"}:
        perception_share = get_package_share_directory("robot_perception")
        perception_prefix = get_package_prefix("robot_perception")
        workspace_root = os.path.dirname(os.path.dirname(perception_prefix))
        gpu_python = os.path.join(workspace_root, ".venv", "bin", "python")
        model_path = LaunchConfiguration("yolo_model_path").perform(context).strip()
        if yolo_backend == "yolox":
            if not os.path.isfile(gpu_python):
                raise RuntimeError(
                    f"YOLOX Python environment was not found at {gpu_python}"
                )
            if not model_path:
                model_path = os.path.join(
                    perception_share,
                    "models",
                    "object_detection_yolox_m_20260918.onnx",
                )
            yolo_nodes.append(Node(
                package="robot_perception",
                executable="yolo_node",
                parameters=[{
                    "image_topic": LaunchConfiguration("yolo_image_topic").perform(context),
                    "model_path": model_path,
                    "confidence": float(
                        LaunchConfiguration("yolo_confidence").perform(context)
                    ),
                    "nms_threshold": float(
                        LaunchConfiguration("yolo_nms_iou").perform(context)
                    ),
                    "class_filter": LaunchConfiguration("yolo_class_filter").perform(context).strip(),
                    "opencv_num_threads": int(
                        LaunchConfiguration("yolo_opencv_num_threads").perform(context)
                    ),
                }],
                prefix=gpu_python,
                output="screen",
            ))
        elif yolo_backend == "yolo11n_dnn":
            yolo_params_file = LaunchConfiguration("yolo_params_file").perform(context).strip()
            if yolo_params_file:
                if not os.path.isfile(yolo_params_file):
                    raise RuntimeError(f"YOLO parameter file was not found at {yolo_params_file}")
                yolo_node_parameters = [yolo_params_file]
            else:
                if not model_path:
                    model_path = os.path.join(
                        workspace_root,
                        "verification",
                        "yolo11n_same_frame_20261008",
                        "weights",
                        "yolo11n.onnx",
                    )
                if not os.path.isfile(model_path):
                    raise RuntimeError(f"YOLO11n ONNX model was not found at {model_path}")
                yolo_node_parameters = [{
                    "image_topic": LaunchConfiguration("yolo_image_topic").perform(context),
                    "model_path": model_path,
                    "confidence": float(
                        LaunchConfiguration("yolo_confidence").perform(context)
                    ),
                    "nms_iou": float(LaunchConfiguration("yolo_nms_iou").perform(context)),
                    "imgsz": int(LaunchConfiguration("yolo_imgsz").perform(context)),
                    "class_filter": LaunchConfiguration("yolo_class_filter").perform(context).strip(),
                    "opencv_num_threads": int(
                        LaunchConfiguration("yolo_opencv_num_threads").perform(context)
                    ),
                }]
            yolo_nodes.append(Node(
                package="robot_perception",
                executable="yolo11n_node",
                parameters=yolo_node_parameters,
                additional_env={
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                output="screen",
            ))
        else:
            raise ValueError(
                "yolo_backend must be 'yolox' or 'yolo11n_dnn'; "
                f"received {yolo_backend!r}"
            )

    return [
        # Select NVIDIA for Mesa D3D12 rendering under WSLg.
        SetEnvironmentVariable("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA"),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            [
                os.path.join(simulation_share, "models"),
                ":",
                os.path.join(assets_share, "assets"),
                ":",
                os.path.dirname(description_share),
                ":",
                EnvironmentVariable("GAZEBO_MODEL_PATH", default_value=""),
            ],
        ),
        SetEnvironmentVariable(
            "GAZEBO_PLUGIN_PATH",
            [
                "/opt/ros/humble/lib:",
                EnvironmentVariable("GAZEBO_PLUGIN_PATH", default_value=""),
            ],
        ),
        gazebo_server,
        gazebo_client,
        state_publisher,
        spawn_model,
        ackermann_command_adapter,
        visual_odometry,
        suitcase_control_gui,
        *yolo_nodes,
        camera_viewer,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("camera_enabled", default_value="true"),
        DeclareLaunchArgument("visual_odometry_enabled", default_value="true"),
        DeclareLaunchArgument("camera_update_rate", default_value="30.0"),
        DeclareLaunchArgument("camera_viewer_enabled", default_value="true"),
        DeclareLaunchArgument("suitcase_control_gui", default_value="true"),
        DeclareLaunchArgument("yolo_enabled", default_value="false"),
        DeclareLaunchArgument("yolo_backend", default_value="yolox"),
        DeclareLaunchArgument("yolo_model_path", default_value=""),
        DeclareLaunchArgument("yolo_params_file", default_value=""),
        DeclareLaunchArgument("yolo_image_topic", default_value="/sim_camera/color/image_raw"),
        DeclareLaunchArgument("yolo_confidence", default_value="0.35"),
        DeclareLaunchArgument("yolo_nms_iou", default_value="0.5"),
        DeclareLaunchArgument("yolo_imgsz", default_value="640"),
        DeclareLaunchArgument("yolo_class_filter", default_value=""),
        DeclareLaunchArgument("yolo_opencv_num_threads", default_value="8"),
        DeclareLaunchArgument("camera_view_topic", default_value="/sim_camera/color/image_raw"),
        DeclareLaunchArgument("world", default_value=""),
        DeclareLaunchArgument("start_paused", default_value="false"),
        DeclareLaunchArgument("robot_spawn_x", default_value="0.0"),
        DeclareLaunchArgument("robot_spawn_y", default_value="0.0"),
        DeclareLaunchArgument("robot_spawn_z", default_value="0.019"),
        DeclareLaunchArgument("robot_spawn_yaw", default_value="0.0"),
        OpaqueFunction(function=_build_simulation),
    ])
