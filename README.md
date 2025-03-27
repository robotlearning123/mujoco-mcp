好的，这是我们根据之前讨论和检查确定的最新版本 **MuJoCo-MCP 需求文档 V1.1** 全文：

---

## MuJoCo-MCP (MuJoCo Motion Control Primitives) 需求文档

**版本:** 1.1
**日期:** 2023年10月27日 (根据对话内容确定，实际日期为当前)

**1. 引言**

*   **1.1 目的:** 本文档旨在明确定义 `MuJoCo-MCP` Python 库的功能、设计要求、性能指标和约束条件。该库旨在为 MuJoCo 物理模拟器提供一个高层次、面向任务的机器人运动控制接口，简化复杂控制策略（如阻抗控制、操作空间控制）的实现和应用。
*   **1.2 背景:** 当前直接使用 `mujoco` 官方 Python 包进行机器人控制需要用户深入了解 MuJoCo 的底层 API 和机器人动力学知识。`MuJoCo-MCP` 的目标是提供一个类似 `H-Freax/PybulletMCP` 的抽象层，但基于现代 `mujoco` 包构建，充分利用其高性能和精确的物理模拟及动力学计算能力。
*   **1.3 范围:**
    *   提供一个封装 `mujoco.MjModel` 和 `mujoco.MjData` 的机器人接口层。
    *   实现标准的关节空间和笛卡尔空间控制律（如阻抗控制、操作空间控制）。
    *   提供用于生成平滑运动轨迹的工具。
    *   提供易于使用的高级运动指令（控制原语）。
    *   提供必要的数学和变换工具函数。
    *   包含示例代码和清晰的文档（含 API、坐标系、单位说明）。
*   **1.4 定义与缩写:**
    *   **MuJoCo:** Multi-Joint dynamics with Contact 物理模拟引擎。
    *   **`mujoco` package:** MuJoCo 官方 Python 绑定库。
    *   **MCP:** Motion Control Primitives (运动控制原语)。
    *   **MJCF:** MuJoCo 原生模型描述格式 (MuJoCo XML Format)。
    *   **URDF:** 通用机器人描述格式 (Unified Robot Description Format)。
    *   **`MjModel`:** MuJoCo 的静态模型数据结构 (`mujoco.MjModel`)。
    *   **`MjData`:** MuJoCo 的动态仿真状态数据结构 (`mujoco.MjData`)。
    *   **EE:** End-Effector (末端执行器)，推荐使用 MJCF 中的 `site` 来定义。
    *   **FK:** Forward Kinematics (正向运动学)。
    *   **Jacobian:** 雅可比矩阵。
    *   **Impedance Control:** 阻抗控制。
    *   **OSC:** Operational Space Control (操作空间控制)。
    *   **`data.qpos`:** MuJoCo `MjData` 中的关节位置向量。
    *   **`data.qvel`:** MuJoCo `MjData` 中的关节速度向量。
    *   **`data.ctrl`:** MuJoCo `MjData` 中定义的执行器控制信号输入。
    *   **`data.qfrc_applied`:** MuJoCo `MjData` 中施加于自由度的广义力（用于直接力矩控制）。
    *   **`data.sensordata`:** MuJoCo `MjData` 中存储传感器读数的数组。
    *   **`data.qfrc_bias`:** MuJoCo `MjData` 中由 `mj_inverse` 计算得到的偏置力（科里奥利力 C + 重力 G）。
    *   **Site:** MuJoCo 模型中用于标记位置和方向的轻量级元素，常用于定义 EE 坐标系。
    *   **SI Units:** 国际单位制（MuJoCo 默认使用）。

**2. 总体要求**

*   **2.1 核心目标:** 在 MuJoCo 模拟器之上构建一个易于使用、高效且功能强大的机器人控制抽象层。
*   **2.2 设计原则:**
    *   **模块化:** 功能模块（接口、控制器、原语、轨迹、工具）清晰分离。
    *   **易用性:** 简洁、直观、Pythonic 的 API。
    *   **效率:** 最大限度利用 MuJoCo 内建 C 语言优化函数（通过 `mujoco` 包调用）。
    *   **明确性:** 清晰标注函数、控制器所使用的坐标系和物理单位。
*   **2.3 模拟器依赖:** 明确依赖 `mujoco` 官方 Python 包 (例如, `>= 3.1.3` 或最新稳定版)。
*   **2.4 控制接口:** 必须明确支持并区分两种主要的 MuJoCo 控制输入方式：
    *   通过 `data.ctrl` 控制 MJCF 文件中定义的执行器。其效果取决于 MJCF 中执行器的类型 (`<motor>`, `<position>`, `<velocity>`, `<general>`) 和参数。
    *   通过 `data.qfrc_applied` 直接施加广义力矩。这是实现自定义力矩控制律（如阻抗控制、OSC）的**首选方式**，因为它绕过了内置的执行器模型，直接应用计算出的力矩。
    库的 API 应允许用户了解控制器的输出目标（`ctrl` 或 `qfrc_applied`），高级控制器（阻抗/OSC）默认应输出力矩到 `qfrc_applied`。
*   **2.5 机器人模型:** 主要支持 MJCF 格式。应能通过 MuJoCo 的内建编译器加载 URDF 文件，但用户需被告知转换可能带来的语义差异。

**3. 功能需求**

*   **3.1 机器人接口模块 (`robot_interface.py`)**
    *   FR1.1: 能够从 MJCF 文件路径或字符串加载 `mujoco.MjModel`。
    *   FR1.2: 能够从 URDF 文件路径加载模型（利用 `mujoco.MjModel.from_xml_path`）。
    *   FR1.3: 创建并管理 `mujoco.MjData` 实例。
    *   FR1.4: 提供获取当前关节位置 (`data.qpos`) 的接口。
    *   FR1.5: 提供获取当前关节速度 (`data.qvel`) 的接口。
    *   FR1.6: 提供获取指定末端执行器 **Site**（通过名称指定）的**世界坐标系**位姿的接口。
        *   位置：`data.site_xpos[site_id]`
        *   方向：`data.site_xmat[site_id]` (3x3 旋转矩阵) 或转换为四元数。
        *   *依赖：* 调用此接口前必须已执行 `mujoco.mj_kinematics(model, data)` 或 `mujoco.mj_forward(model, data)`。
    *   FR1.7: 提供获取指定末端执行器 **Site** 的**世界坐标系**空间速度（6D 向量：线速度、角速度）的接口。
        *   *实现：* 使用 `mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel_buffer, 0)`。`vel_buffer` 是预分配的 6x1 NumPy 数组。`0` 表示结果在世界坐标系。
    *   FR1.8: 提供计算并获取指定末端执行器 **Site** 的**世界坐标系**雅可比矩阵的接口。
        *   *实现：* 使用 `mujoco.mj_jacSite(model, data, jacp_buffer, jacr_buffer, site_id)`。`jacp_buffer` (3xNv) 和 `jacr_buffer` (3xNv) 是预分配的 NumPy 数组，存储平移和旋转雅可比。组合成 6xNv 雅可比矩阵。
        *   *依赖：* 调用此接口前必须已执行包含运动学计算的步骤。
    *   FR1.9: 提供计算并获取机器人（完整系统的）质量矩阵 `M` 的接口。
        *   *实现：* 使用 `mujoco.mj_fullM(model, M_buffer, data.qM)` (注意: 根据最新 API，质量矩阵在 `data.qM` 中，需要 `mj_forward` 计算)。`M_buffer` 是预分配的 Nv x Nv NumPy 数组用于拷贝 `data.qM`。或者，如果仅需 `M` 用于计算，可以直接使用 `data.qM`。
    *   FR1.10: 提供计算并获取偏置力向量（科里奥利力 C + 重力 G）`data.qfrc_bias` 的接口。
        *   *实现：* 调用 `mujoco.mj_inverse(model, data)` 后，从 `data.qfrc_bias` 读取。
    *   FR1.11: 提供设置 `data.ctrl` 控制信号的接口。
    *   FR1.12: 提供设置 `data.qfrc_applied` 广义力矩的接口。
    *   FR1.13: 提供执行单个 MuJoCo 模拟步进 (`mujoco.mj_step(model, data)`) 的接口。
    *   FR1.14: 提供根据名称获取 MuJoCo 对象 ID 的辅助函数。
        *   *实现：* 使用现代 API，例如 `model.body(name).id`, `model.site(name).id`, `model.joint(name).id`, `model.actuator(name).id`, `model.sensor(name).id`。
    *   FR1.15: 提供读取指定传感器（通过名称指定）数据的接口。
        *   *实现：* 通过 `model.sensor(name).id` 获取传感器 ID，通过 `model.sensor(id).adr` 和 `model.sensor(id).dim` 确定在 `data.sensordata` 中的起始地址和维度。
        *   *依赖：* 可能需要在 `mj_step` 之后，或在需要基于约束力的传感器时调用 `mujoco.mj_rnePostConstraint(model, data)` 之后读取。

*   **3.2 控制器模块 (`controllers/`)**
    *   FR2.1: 定义一个抽象基类 `BaseController`，规定控制器接口（例如 `compute_torque(self, model, data)` 方法返回计算出的力矩）。
    *   FR2.2: **关节空间阻抗控制器 (`JointImpedanceController`)**
        *   输入：目标关节位置 `q_des`、目标关节速度 `qd_des`（可选，默认为0）、关节刚度 `Kp` (Nv x Nv 或 Nv-dim vector)、关节阻尼 `Kd` (Nv x Nv 或 Nv-dim vector)。
        *   输出：计算得到的关节力矩 `tau` (Nv-dim vector)。
        *   实现：`tau = Kp @ (q_des - data.qpos) + Kd @ (qd_des - data.qvel) + data.qfrc_bias`。其中 `data.qfrc_bias` 通过 `robot_interface` 获取（内部需调用 `mj_inverse`）。
        *   **目标:** 输出的 `tau` 默认用于 `data.qfrc_applied`。
    *   FR2.3: **笛卡尔空间阻抗控制器 (`CartesianImpedanceController`)**
        *   输入：目标 EE Site 位姿 (`pose_des`: 位置向量, 旋转矩阵/四元数)、目标 EE Site 世界坐标系速度 (`vel_des` 6D 向量，可选，默认为0)、笛卡尔刚度 `Kp_cart` (6x6 matrix 或 6-dim vector)、笛卡尔阻尼 `Kd_cart` (6x6 matrix 或 6-dim vector)。
        *   输出：计算得到的关节力矩 `tau` (Nv-dim vector)。
        *   实现：
            1.  获取当前 EE Site 位姿 (`pose_curr`) 和世界坐标系速度 (`vel_curr`) (使用 FR1.6, FR1.7)。
            2.  计算位姿误差 (6D):
                *   位置误差: `err_p = pose_des.pos - pose_curr.pos`
                *   方向误差: `err_o = compute_orientation_error(pose_des.rot, pose_curr.rot)` (例如使用 `scipy.spatial.transform.Rotation.from_matrix(R_des @ R_curr.T).as_rotvec()`)。
            3.  计算笛卡尔空间期望力/力矩: `F_des = Kp_cart @ err_pose + Kd_cart @ (vel_des - vel_curr)`。
            4.  获取世界坐标系雅可比 `J` (6xNv) (使用 FR1.8)。
            5.  获取偏置力 `data.qfrc_bias` (使用 FR1.10)。
            6.  计算关节力矩: `tau = J.T @ F_des + data.qfrc_bias`。
            7.  **奇异点处理:** 在计算 `tau` 时应加入阻尼最小二乘法 (Damped Least Squares) 或类似机制来处理雅可比奇异或接近奇异的情况: `J.T @ (J @ J.T + damping * I)^-1 @ F_des` 或 `J.T @ F_des` 中应用阻尼。
        *   **目标:** 输出的 `tau` 默认用于 `data.qfrc_applied`。
    *   FR2.4: **操作空间控制器 (`OperationalSpaceController`)**
        *   输入：任务空间的目标（例如 EE Site 期望加速度 `x_ddot_des` 或期望力 `F_des`）、控制器增益等。
        *   输出：计算得到的关节力矩 `tau`。
        *   实现 (示例 - 任务空间逆动力学):
            1.  获取雅可比 `J` (FR1.8)，质量矩阵 `M` (使用 `data.qM`, FR1.9)，偏置力 `data.qfrc_bias` (FR1.10)。
            2.  (可选) 计算操作空间质量矩阵 `Lambda = (J @ solve(M, J.T))^-1` (使用 `numpy.linalg.solve` 或类似方法求解 `M * X = J.T` 来避免直接求逆 `M`)。
            3.  (可选) 计算 `J_dot @ qvel` 项（可以通过有限差分或 MuJoCo 的 `mj_jac*dot` 相关函数）。
            4.  根据具体的 OSC 公式计算任务空间力 `F_task`。例如 `F_task = Lambda @ (x_ddot_des - J_dot @ qvel) + Mu_task` (Mu_task 是操作空间偏置力)。
            5.  计算关节力矩: `tau = J.T @ F_task + data.qfrc_bias`。 (根据公式变种，`data.qfrc_bias` 可能部分或全部包含在 `F_task` 的推导中)。
            6.  **初始范围:** 优先实现基于 `data.qfrc_bias` 进行补偿的基础版本。空空间控制 (`N @ tau_nullspace`) 可作为后续扩展。
            7.  **奇异点处理:** 必须加入阻尼或其他正则化方法处理 `Lambda` 的计算或雅可比转置的应用，例如 `Lambda = (J @ solve(M, J.T) + damping * I)^-1`。
        *   **目标:** 输出的 `tau` 默认用于 `data.qfrc_applied`。

*   **3.3 控制原语模块 (`primitives/`)**
    *   FR3.1: 定义抽象基类 `BasePrimitive`。
    *   FR3.2: `MoveJointTo`: 使用 `Trajectory` 生成关节轨迹，循环调用 `JointImpedanceController` 等计算 `tau` 并通过 `robot_interface.set_qfrc_applied()` 应用。
    *   FR3.3: `MovePoseCartesian`: 使用 `Trajectory` 生成笛卡尔轨迹，循环调用 `CartesianImpedanceController` 或 `OSC` 计算 `tau` 并通过 `robot_interface.set_qfrc_applied()` 应用。
    *   FR3.4: 原语应处理执行状态、成功/失败条件（目标容差、超时、检测到的异常）。

*   **3.4 轨迹生成模块 (`trajectory/`)**
    *   FR4.1: 提供生成器，用于创建关节空间和笛卡尔空间（位置和方向）的平滑轨迹。
    *   FR4.2: 支持基于时间的多项式插值（如最小加加速度/Minimum Jerk）。
    *   FR4.3: 输入：起始点、目标点、可选的中间航点、总时长或各段时长。
    *   FR4.4: 输出：在给定时间 `t` 处的期望位置、速度和加速度。

*   **3.5 工具模块 (`utils/`)**
    *   FR5.1: `kinematics.py`: 封装 FR1.6, FR1.7, FR1.8 等接口调用。
    *   FR5.2: `transformations.py`: 姿态转换 (`scipy.spatial.transform.Rotation`)、位姿运算、雅可比坐标系转换（如果需要非世界坐标系）。
    *   FR5.3: `mujoco_utils.py`: 封装 FR1.14 (ID 查找) 等常用操作。

*   **3.6 异常处理 (`exceptions.py`)**
    *   FR6.1: 定义库特定的异常类，用于指示配置错误、模拟错误、控制失败等。

*   **3.7 示例 (`examples/`)**
    *   FR7.1: 提供脚本展示如何加载标准机器人模型（例如，来自 `mujoco-menagerie` 的 Panda）。
    *   FR7.2: 演示如何使用 `RobotInterface` 获取状态和发送命令。
    *   FR7.3: 演示如何实例化和使用各种控制器（关节阻抗、笛卡尔阻抗、OSC）。
    *   FR7.4: 演示如何使用运动原语（`MoveJointTo`, `MovePoseCartesian`）。
    *   FR7.5: 示例应包含使用 `mujoco-viewer` 进行可视化的代码。

*   **3.8 测试 (`tests/`)**
    *   FR8.1: 为 `utils` 模块中的数学和变换函数提供单元测试。
    *   FR8.2: 为 `trajectory` 模块的轨迹生成提供单元测试。
    *   FR8.3: 为 `controllers` 提供集成测试（可能需要运行短时模拟来验证输出）。
    *   FR8.4: 为 `primitives` 提供集成测试。

**4. 非功能需求**

*   **NF1. 性能:** 控制器计算频率目标 > 100Hz (简单) / > 500Hz (理想)。避免 Python 循环内低效操作。
*   **NF2. 易用性:** 直观 API，清晰文档、示例，明确错误信息。
*   **NF3. 可靠性:** 库稳定性，适当处理 MuJoCo 错误。**明确要求在笛卡尔控制器 (FR2.3, FR2.4) 中实现雅可比奇异点处理机制 (如 Damped Least Squares)。**
*   **NF4. 可维护性与可扩展性:** 模块化，PEP 8，类型提示，良好注释。
*   **NF5. 文档:**
    *   API 参考 (可用 `sphinx`)。
    *   教程和概念解释。
    *   **明确要求文档中标注所有接口和控制器使用的物理量单位（默认为 SI 单位）。**
    *   **明确要求文档中标注坐标系约定（例如，EE 位姿/速度/雅可比默认在世界坐标系）。**

**5. 依赖项**

*   **D1:** Python (例如 >= 3.8)
*   **D2:** `mujoco` (例如 >= 3.1.3 或最新稳定版)
*   **D3:** `numpy`
*   **D4:** `scipy` (用于变换和插值)
*   **D5:** (可选，用于示例) `mujoco-viewer`

**6. 目标环境**

*   **E1:** 操作系统：Linux, macOS, Windows (所有 `mujoco` 支持的平台)。

**7. 未来考虑 (可选)**
*   更复杂的 OSC (优先级、约束)、力控制原语、运动规划接口、特定机器人封装。

---

**V1.1 更新说明:**

*   更新了 FR1.9，指明质量矩阵 `M` 应从 `data.qM` 获取（需要 `mj_forward`），而不是通过 `mj_fullM` 计算填充外部 buffer。
*   在 FR2.3 和 FR2.4 中细化了奇异点处理的实现方式示例 (DLS)。
*   在 FR2.4 中建议使用 `solve(M, J.T)` 替代 `inv(M) @ J.T` 以提高数值稳定性。
*   根据最新的 MuJoCo 文档和实践对措辞进行了微调，确保 API 引用的准确性。
*   保持了 V1.0 的所有核心需求和结构。
