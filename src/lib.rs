#![forbid(unsafe_code)]

//! Deterministic Dubins path planning using integer arithmetic only.
//!
//! # Units
//! - Positions and lengths use fixed-point integers with `SCALE = 1000`.
//! - Angles are in milliradians (`Angle`), where 1000 = 1 rad.
//! - The minimum turning radius (`rho`) uses the same fixed-point scale.
//!
//! This keeps all computations deterministic across architectures.

use deterministic_trigonometry::DTrig;

/// Fixed-point scale for distances and unitless values.
pub const SCALE: i64 = 1000;

/// Angle in milliradians (radians * 1000).
pub type Angle = i32;

/// Fixed-point distance or unitless value (scaled by `SCALE`).
pub type Fixed = i64;

/// 2 * PI in milliradians (approx).
pub const TAU_MRAD: Angle = 6283;

/// PI in milliradians (approx).
pub const PI_MRAD: Angle = 3142;

/// PI / 2 in milliradians (approx).
pub const HALF_PI_MRAD: Angle = 1571;

/// Circular manifold (center + radius).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircleManifold {
    pub center: FixedVec2,
    pub radius: Fixed,
}

/// Tangent direction relative to the circle (clockwise or counterclockwise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TangentDirection {
    Clockwise,
    Counterclockwise,
}

/// Options for manifold search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifoldSearchOptions {
    /// Angle step in milliradians.
    pub angle_step: Angle,
    /// Whether to consider clockwise tangents on the start circle.
    pub start_allow_clockwise: bool,
    /// Whether to consider counterclockwise tangents on the start circle.
    pub start_allow_counterclockwise: bool,
    /// Whether to consider clockwise tangents on the end circle.
    pub end_allow_clockwise: bool,
    /// Whether to consider counterclockwise tangents on the end circle.
    pub end_allow_counterclockwise: bool,
}

impl Default for ManifoldSearchOptions {
    fn default() -> Self {
        Self {
            angle_step: 10,
            start_allow_clockwise: true,
            start_allow_counterclockwise: true,
            end_allow_clockwise: true,
            end_allow_counterclockwise: true,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FixedVec2 {
    pub x: Fixed,
    pub y: Fixed,
}

impl FixedVec2 {
    pub fn new(x: Fixed, y: Fixed) -> Self {
        Self { x, y }
    }

    pub fn distance(self, other: Self) -> Fixed {
        let dx = (other.x - self.x) as i128;
        let dy = (other.y - self.y) as i128;
        isqrt_i128(dx * dx + dy * dy)
    }
}

/// A pose in 2D with a heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pose {
    /// X position in fixed-point units.
    pub x: Fixed,
    /// Y position in fixed-point units.
    pub y: Fixed,
    /// Heading in milliradians.
    pub heading: Angle,
}

impl Pose {
    /// Creates a pose from fixed-point inputs.
    pub fn new_fixed(x: Fixed, y: Fixed, heading: Angle) -> Self {
        Self { x, y, heading: normalize_angle(heading) }
    }

    /// Creates a pose from integer units (scaled internally by `SCALE`).
    pub fn new_units(x: i64, y: i64, heading: Angle) -> Self {
        Self::new_fixed(x.saturating_mul(SCALE), y.saturating_mul(SCALE), heading)
    }
}

/// Segment type for a Dubins path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    Left,
    Straight,
    Right,
}

/// Dubins path family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathType {
    LSL,
    RSR,
    LSR,
    RSL,
    RLR,
    LRL,
}

/// One segment in a Dubins path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DubinsSegment {
    /// Segment type.
    pub kind: SegmentType,
    /// Segment length in fixed-point units.
    pub length: Fixed,
}

/// A deterministic Dubins path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DubinsPath {
    /// Start pose for this path.
    pub start: Pose,
    /// Minimum turning radius in fixed-point units.
    pub rho: Fixed,
    /// Segment list in order.
    pub segments: [DubinsSegment; 3],
    /// Path family.
    pub path_type: PathType,
    /// Total path length in fixed-point units.
    pub total_length: Fixed,
}

/// Errors for Dubins planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DubinsError {
    /// Turning radius must be positive.
    InvalidRadius,
    /// Angle step must be positive.
    InvalidStep,
    /// No valid path found due to numeric issues.
    NoPath,
}

/// Deterministic planner context with cached trigonometry tables.
pub struct DubinsContext {
    dtrig: DTrig,
}

impl Default for DubinsContext {
    fn default() -> Self {
        Self::new()
    }
}

impl DubinsContext {
    /// Creates a new deterministic context.
    pub fn new() -> Self {
        Self { dtrig: DTrig::initialize() }
    }

    /// Computes the shortest Dubins path between two poses.
    pub fn shortest_path(
        &self,
        start: Pose,
        end: Pose,
        rho: Fixed,
    ) -> Result<DubinsPath, DubinsError> {
        let mut paths = self.all_paths(start, end, rho)?;
        paths.sort_by_key(|path| path.total_length);
        paths.into_iter().next().ok_or(DubinsError::NoPath)
    }

    /// Computes all feasible Dubins paths between two poses.
    pub fn all_paths(
        &self,
        start: Pose,
        end: Pose,
        rho: Fixed,
    ) -> Result<Vec<DubinsPath>, DubinsError> {
        if rho <= 0 {
            return Err(DubinsError::InvalidRadius);
        }

        let (alpha, beta, d) = normalized_inputs(start, end, rho, &self.dtrig);

        let mut candidates = Vec::new();
        if let Some(candidate) = dubins_lsl(alpha, beta, d, &self.dtrig) {
            candidates.push(candidate);
        }
        if let Some(candidate) = dubins_rsr(alpha, beta, d, &self.dtrig) {
            candidates.push(candidate);
        }
        if let Some(candidate) = dubins_lsr(alpha, beta, d, &self.dtrig) {
            candidates.push(candidate);
        }
        if let Some(candidate) = dubins_rsl(alpha, beta, d, &self.dtrig) {
            candidates.push(candidate);
        }
        if let Some(candidate) = dubins_rlr(alpha, beta, d, &self.dtrig) {
            candidates.push(candidate);
        }
        if let Some(candidate) = dubins_lrl(alpha, beta, d, &self.dtrig) {
            candidates.push(candidate);
        }

        if candidates.is_empty() {
            return Err(DubinsError::NoPath);
        }

        Ok(candidates
            .into_iter()
            .map(|candidate| candidate.to_path(start, rho))
            .collect())
    }

    /// Samples the path at a given arc-length.
    pub fn sample(&self, path: &DubinsPath, distance: Fixed) -> Pose {
        let mut remaining = distance.clamp(0, path.total_length);
        let mut pose = path.start;

        for segment in path.segments.iter() {
            if remaining <= 0 {
                break;
            }
            let step = remaining.min(segment.length);
            pose = propagate_segment(pose, *segment, step, path.rho, &self.dtrig);
            remaining -= step;
        }

        pose
    }

    /// Shortest path from a pose to a tangent point on a circle (grid search).
    pub fn shortest_path_pose_to_circle_grid(
        &self,
        start: Pose,
        circle: CircleManifold,
        rho: Fixed,
        options: ManifoldSearchOptions,
    ) -> Result<PoseToCircleResult, DubinsError> {
        if circle.radius <= 0 || rho <= 0 {
            return Err(DubinsError::InvalidRadius);
        }
        if options.angle_step <= 0 {
            return Err(DubinsError::InvalidStep);
        }

        let mut best: Option<PoseToCircleResult> = None;
        let step = options.angle_step as usize;
        for angle in (0..TAU_MRAD).step_by(step) {
            if options.end_allow_clockwise {
                best = select_best_pose_to_circle(
                    best,
                    self,
                    start,
                    circle,
                    angle,
                    TangentDirection::Clockwise,
                    rho,
                );
            }
            if options.end_allow_counterclockwise {
                best = select_best_pose_to_circle(
                    best,
                    self,
                    start,
                    circle,
                    angle,
                    TangentDirection::Counterclockwise,
                    rho,
                );
            }
        }
        best.ok_or(DubinsError::NoPath)
    }

    /// Shortest path from a circle tangent start to a pose (grid search).
    pub fn shortest_path_circle_to_pose_grid(
        &self,
        circle: CircleManifold,
        end: Pose,
        rho: Fixed,
        options: ManifoldSearchOptions,
    ) -> Result<CircleToPoseResult, DubinsError> {
        if circle.radius <= 0 || rho <= 0 {
            return Err(DubinsError::InvalidRadius);
        }
        if options.angle_step <= 0 {
            return Err(DubinsError::InvalidStep);
        }

        let mut best: Option<CircleToPoseResult> = None;
        let step = options.angle_step as usize;
        for angle in (0..TAU_MRAD).step_by(step) {
            if options.start_allow_clockwise {
                best = select_best_circle_to_pose(
                    best,
                    self,
                    circle,
                    end,
                    angle,
                    TangentDirection::Clockwise,
                    rho,
                );
            }
            if options.start_allow_counterclockwise {
                best = select_best_circle_to_pose(
                    best,
                    self,
                    circle,
                    end,
                    angle,
                    TangentDirection::Counterclockwise,
                    rho,
                );
            }
        }
        best.ok_or(DubinsError::NoPath)
    }

    /// Shortest path between two circular manifolds (tangent-to-tangent, grid search).
    pub fn shortest_path_circle_to_circle_grid(
        &self,
        start_circle: CircleManifold,
        end_circle: CircleManifold,
        rho: Fixed,
        options: ManifoldSearchOptions,
    ) -> Result<CircleToCircleResult, DubinsError> {
        if start_circle.radius <= 0 || end_circle.radius <= 0 || rho <= 0 {
            return Err(DubinsError::InvalidRadius);
        }
        if options.angle_step <= 0 {
            return Err(DubinsError::InvalidStep);
        }

        let mut best: Option<CircleToCircleResult> = None;
        let step = options.angle_step as usize;
        let directions = [TangentDirection::Clockwise, TangentDirection::Counterclockwise];

        for start_angle in (0..TAU_MRAD).step_by(step) {
            for end_angle in (0..TAU_MRAD).step_by(step) {
                for start_dir in directions {
                    if !start_dir_allowed(start_dir, options) {
                        continue;
                    }
                    for end_dir in directions {
                        if !end_dir_allowed(end_dir, options) {
                            continue;
                        }
                        best = select_best_circle_to_circle(
                            best,
                            self,
                            start_circle,
                            end_circle,
                            start_angle,
                            end_angle,
                            start_dir,
                            end_dir,
                            rho,
                        );
                    }
                }
            }
        }
        best.ok_or(DubinsError::NoPath)
    }

    /// Shortest path from a pose to a tangent point on a circle (analytic tangents).
    pub fn shortest_path_pose_to_circle_analytic(
        &self,
        start: Pose,
        circle: CircleManifold,
        rho: Fixed,
        options: ManifoldSearchOptions,
    ) -> Result<PoseToCircleResult, DubinsError> {
        if circle.radius <= 0 || rho <= 0 {
            return Err(DubinsError::InvalidRadius);
        }
        if !options.end_allow_clockwise && !options.end_allow_counterclockwise {
            return Err(DubinsError::NoPath);
        }

        let start_point = FixedVec2::new(start.x, start.y);
        let angles = tangent_angles_point_circle(start_point, circle, &self.dtrig);
        if angles.is_empty() {
            return self.shortest_path_pose_to_circle_grid(start, circle, rho, options);
        }
        let mut best: Option<PoseToCircleResult> = None;
        for angle in angles {
            if options.end_allow_clockwise {
                best = select_best_pose_to_circle(
                    best,
                    self,
                    start,
                    circle,
                    angle,
                    TangentDirection::Clockwise,
                    rho,
                );
            }
            if options.end_allow_counterclockwise {
                best = select_best_pose_to_circle(
                    best,
                    self,
                    start,
                    circle,
                    angle,
                    TangentDirection::Counterclockwise,
                    rho,
                );
            }
        }
        best.ok_or(DubinsError::NoPath)
    }

    /// Shortest path from a circle tangent start to a pose (analytic tangents).
    pub fn shortest_path_circle_to_pose_analytic(
        &self,
        circle: CircleManifold,
        end: Pose,
        rho: Fixed,
        options: ManifoldSearchOptions,
    ) -> Result<CircleToPoseResult, DubinsError> {
        if circle.radius <= 0 || rho <= 0 {
            return Err(DubinsError::InvalidRadius);
        }
        if !options.start_allow_clockwise && !options.start_allow_counterclockwise {
            return Err(DubinsError::NoPath);
        }

        let end_point = FixedVec2::new(end.x, end.y);
        let angles = tangent_angles_point_circle(end_point, circle, &self.dtrig);
        if angles.is_empty() {
            return self.shortest_path_circle_to_pose_grid(circle, end, rho, options);
        }
        let mut best: Option<CircleToPoseResult> = None;
        for angle in angles {
            if options.start_allow_clockwise {
                best = select_best_circle_to_pose(
                    best,
                    self,
                    circle,
                    end,
                    angle,
                    TangentDirection::Clockwise,
                    rho,
                );
            }
            if options.start_allow_counterclockwise {
                best = select_best_circle_to_pose(
                    best,
                    self,
                    circle,
                    end,
                    angle,
                    TangentDirection::Counterclockwise,
                    rho,
                );
            }
        }
        best.ok_or(DubinsError::NoPath)
    }

    /// Shortest path between two circular manifolds (analytic tangents).
    pub fn shortest_path_circle_to_circle_analytic(
        &self,
        start_circle: CircleManifold,
        end_circle: CircleManifold,
        rho: Fixed,
        options: ManifoldSearchOptions,
    ) -> Result<CircleToCircleResult, DubinsError> {
        if start_circle.radius <= 0 || end_circle.radius <= 0 || rho <= 0 {
            return Err(DubinsError::InvalidRadius);
        }
        if (!options.start_allow_clockwise && !options.start_allow_counterclockwise)
            || (!options.end_allow_clockwise && !options.end_allow_counterclockwise)
        {
            return Err(DubinsError::NoPath);
        }

        let tangents = circle_circle_tangents(start_circle, end_circle);
        if tangents.is_empty() {
            return self.shortest_path_circle_to_circle_grid(
                start_circle,
                end_circle,
                rho,
                options,
            );
        }
        let mut best: Option<CircleToCircleResult> = None;

        for tangent in tangents {
            if !start_dir_allowed(tangent.start_direction, options)
                || !end_dir_allowed(tangent.end_direction, options)
            {
                continue;
            }

            let line_dx = tangent.end_point.x - tangent.start_point.x;
            let line_dy = tangent.end_point.y - tangent.start_point.y;
            let start_heading = atan2_mrad(line_dy, line_dx, &self.dtrig);
            let end_heading = start_heading;
            let start_pose = Pose::new_fixed(tangent.start_point.x, tangent.start_point.y, start_heading);
            let end_pose = Pose::new_fixed(tangent.end_point.x, tangent.end_point.y, end_heading);
            let start_radial = FixedVec2::new(
                tangent.start_point.x - start_circle.center.x,
                tangent.start_point.y - start_circle.center.y,
            );
            let end_radial = FixedVec2::new(
                tangent.end_point.x - end_circle.center.x,
                tangent.end_point.y - end_circle.center.y,
            );
            let start_angle = atan2_mrad(start_radial.y, start_radial.x, &self.dtrig);
            let end_angle = atan2_mrad(end_radial.y, end_radial.x, &self.dtrig);
            let path = self.shortest_path(start_pose, end_pose, rho).ok();
            if let Some(path) = path {
                let candidate = CircleToCircleResult {
                    path,
                    start_pose,
                    end_pose,
                    start_angle,
                    end_angle,
                    start_direction: tangent.start_direction,
                    end_direction: tangent.end_direction,
                };
                best = match best {
                    None => Some(candidate),
                    Some(current) => {
                        if candidate.path.total_length < current.path.total_length {
                            Some(candidate)
                        } else {
                            Some(current)
                        }
                    }
                };
            }
        }

        best.ok_or(DubinsError::NoPath)
    }
}

/// Result for pose-to-circle searches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoseToCircleResult {
    pub path: DubinsPath,
    pub end_pose: Pose,
    pub end_angle: Angle,
    pub end_direction: TangentDirection,
}

/// Result for circle-to-pose searches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircleToPoseResult {
    pub path: DubinsPath,
    pub start_pose: Pose,
    pub start_angle: Angle,
    pub start_direction: TangentDirection,
}

/// Result for circle-to-circle searches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircleToCircleResult {
    pub path: DubinsPath,
    pub start_pose: Pose,
    pub end_pose: Pose,
    pub start_angle: Angle,
    pub end_angle: Angle,
    pub start_direction: TangentDirection,
    pub end_direction: TangentDirection,
}

#[derive(Clone, Copy, Debug)]
struct CircleTangentCandidate {
    start_point: FixedVec2,
    end_point: FixedVec2,
    start_direction: TangentDirection,
    end_direction: TangentDirection,
}

fn start_dir_allowed(direction: TangentDirection, options: ManifoldSearchOptions) -> bool {
    match direction {
        TangentDirection::Clockwise => options.start_allow_clockwise,
        TangentDirection::Counterclockwise => options.start_allow_counterclockwise,
    }
}

fn end_dir_allowed(direction: TangentDirection, options: ManifoldSearchOptions) -> bool {
    match direction {
        TangentDirection::Clockwise => options.end_allow_clockwise,
        TangentDirection::Counterclockwise => options.end_allow_counterclockwise,
    }
}

fn circle_tangent_pose(
    circle: CircleManifold,
    angle: Angle,
    direction: TangentDirection,
    dtrig: &DTrig,
) -> Pose {
    let point = fixed_from_angle(circle.center, circle.radius, angle, dtrig);
    let heading = match direction {
        TangentDirection::Clockwise => mod2pi(angle as i64 - HALF_PI_MRAD as i64),
        TangentDirection::Counterclockwise => mod2pi(angle as i64 + HALF_PI_MRAD as i64),
    };
    Pose::new_fixed(point.x, point.y, heading)
}

fn tangent_angles_point_circle(
    point: FixedVec2,
    circle: CircleManifold,
    dtrig: &DTrig,
) -> Vec<Angle> {
    let dx = point.x - circle.center.x;
    let dy = point.y - circle.center.y;
    let distance = isqrt_i128((dx as i128) * (dx as i128) + (dy as i128) * (dy as i128));
    if distance == 0 || distance < circle.radius {
        return Vec::new();
    }

    let ratio = scaled_ratio_1000(circle.radius, distance);
    let alpha = match arccos_mrad(ratio, dtrig) {
        Some(value) => value,
        None => return Vec::new(),
    };
    let theta = atan2_mrad(dy, dx, dtrig);

    let mut angles = Vec::new();
    let angle_a = mod2pi(theta as i64 + alpha as i64);
    angles.push(angle_a);
    if alpha != 0 {
        let angle_b = mod2pi(theta as i64 - alpha as i64);
        angles.push(angle_b);
    }
    angles
}

fn scaled_ratio_1000(numer: Fixed, denom: Fixed) -> i64 {
    if denom <= 0 {
        return 0;
    }
    let value = (numer as i128) * 1000 / (denom as i128);
    if value > i64::MAX as i128 {
        i64::MAX
    } else if value < i64::MIN as i128 {
        i64::MIN
    } else {
        value as i64
    }
}

fn tangent_direction(radial: FixedVec2, tangent: FixedVec2) -> TangentDirection {
    let cross = (radial.x as i128) * (tangent.y as i128) - (radial.y as i128) * (tangent.x as i128);
    if cross >= 0 {
        TangentDirection::Counterclockwise
    } else {
        TangentDirection::Clockwise
    }
}

fn circle_circle_tangents(
    start_circle: CircleManifold,
    end_circle: CircleManifold,
) -> Vec<CircleTangentCandidate> {
    let dx = end_circle.center.x - start_circle.center.x;
    let dy = end_circle.center.y - start_circle.center.y;
    let d2 = (dx as i128) * (dx as i128) + (dy as i128) * (dy as i128);
    if d2 == 0 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    let signs = [1i64, -1i64];
    for s in signs {
        let r = end_circle.radius - s * start_circle.radius;
        let h2 = d2 - (r as i128) * (r as i128);
        if h2 < 0 {
            continue;
        }
        let h = isqrt_i128(h2);
        let t_values: &[i64] = if h2 == 0 { &[1] } else { &[-1, 1] };
        for t in t_values {
            let vx_num = (dx as i128) * (r as i128) + (-dy as i128) * (h as i128) * (*t as i128);
            let vy_num = (dy as i128) * (r as i128) + (dx as i128) * (h as i128) * (*t as i128);

            let start_x = start_circle.center.x + div_i128((start_circle.radius as i128) * vx_num, d2);
            let start_y = start_circle.center.y + div_i128((start_circle.radius as i128) * vy_num, d2);
            let end_x = end_circle.center.x + div_i128((end_circle.radius as i128) * vx_num, d2);
            let end_y = end_circle.center.y + div_i128((end_circle.radius as i128) * vy_num, d2);

            let start_point = FixedVec2::new(start_x, start_y);
            let end_point = FixedVec2::new(end_x, end_y);
            let line_dir = FixedVec2::new(end_x - start_x, end_y - start_y);
            if line_dir.x == 0 && line_dir.y == 0 {
                continue;
            }

            let start_radial = FixedVec2::new(start_x - start_circle.center.x, start_y - start_circle.center.y);
            let end_radial = FixedVec2::new(end_x - end_circle.center.x, end_y - end_circle.center.y);
            let start_direction = tangent_direction(start_radial, line_dir);
            let end_direction = tangent_direction(end_radial, line_dir);

            candidates.push(CircleTangentCandidate {
                start_point,
                end_point,
                start_direction,
                end_direction,
            });
        }
    }
    candidates
}

fn div_i128(numer: i128, denom: i128) -> i64 {
    if denom == 0 {
        return 0;
    }
    (numer / denom) as i64
}

fn select_best_pose_to_circle(
    best: Option<PoseToCircleResult>,
    ctx: &DubinsContext,
    start: Pose,
    circle: CircleManifold,
    angle: Angle,
    direction: TangentDirection,
    rho: Fixed,
) -> Option<PoseToCircleResult> {
    let end_pose = circle_tangent_pose(circle, angle, direction, &ctx.dtrig);
    let path = ctx.shortest_path(start, end_pose, rho).ok()?;
    let candidate = PoseToCircleResult {
        path,
        end_pose,
        end_angle: angle,
        end_direction: direction,
    };
    match best {
        None => Some(candidate),
        Some(current) => {
            if candidate.path.total_length < current.path.total_length {
                Some(candidate)
            } else {
                Some(current)
            }
        }
    }
}

fn select_best_circle_to_pose(
    best: Option<CircleToPoseResult>,
    ctx: &DubinsContext,
    circle: CircleManifold,
    end: Pose,
    angle: Angle,
    direction: TangentDirection,
    rho: Fixed,
) -> Option<CircleToPoseResult> {
    let start_pose = circle_tangent_pose(circle, angle, direction, &ctx.dtrig);
    let path = ctx.shortest_path(start_pose, end, rho).ok()?;
    let candidate = CircleToPoseResult {
        path,
        start_pose,
        start_angle: angle,
        start_direction: direction,
    };
    match best {
        None => Some(candidate),
        Some(current) => {
            if candidate.path.total_length < current.path.total_length {
                Some(candidate)
            } else {
                Some(current)
            }
        }
    }
}

fn select_best_circle_to_circle(
    best: Option<CircleToCircleResult>,
    ctx: &DubinsContext,
    start_circle: CircleManifold,
    end_circle: CircleManifold,
    start_angle: Angle,
    end_angle: Angle,
    start_direction: TangentDirection,
    end_direction: TangentDirection,
    rho: Fixed,
) -> Option<CircleToCircleResult> {
    let start_pose = circle_tangent_pose(start_circle, start_angle, start_direction, &ctx.dtrig);
    let end_pose = circle_tangent_pose(end_circle, end_angle, end_direction, &ctx.dtrig);
    let path = ctx.shortest_path(start_pose, end_pose, rho).ok()?;
    let candidate = CircleToCircleResult {
        path,
        start_pose,
        end_pose,
        start_angle,
        end_angle,
        start_direction,
        end_direction,
    };
    match best {
        None => Some(candidate),
        Some(current) => {
            if candidate.path.total_length < current.path.total_length {
                Some(candidate)
            } else {
                Some(current)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub path_type: PathType,
    pub t: i64,
    pub p: i64,
    pub q: i64,
}

impl Candidate {
    fn to_path(self, start: Pose, rho: Fixed) -> DubinsPath {
        let segments = match self.path_type {
            PathType::LSL => [SegmentType::Left, SegmentType::Straight, SegmentType::Left],
            PathType::RSR => [SegmentType::Right, SegmentType::Straight, SegmentType::Right],
            PathType::LSR => [SegmentType::Left, SegmentType::Straight, SegmentType::Right],
            PathType::RSL => [SegmentType::Right, SegmentType::Straight, SegmentType::Left],
            PathType::RLR => [SegmentType::Right, SegmentType::Left, SegmentType::Right],
            PathType::LRL => [SegmentType::Left, SegmentType::Right, SegmentType::Left],
        };

        let values = [self.t, self.p, self.q];
        let mut total_length = 0i64;
        let mut built_segments = [DubinsSegment { kind: SegmentType::Straight, length: 0 }; 3];

        for i in 0..3 {
            let length = scaled_mul(rho, values[i]) / SCALE;
            built_segments[i] = DubinsSegment { kind: segments[i], length };
            total_length = total_length.saturating_add(length);
        }

        DubinsPath {
            start,
            rho,
            segments: built_segments,
            path_type: self.path_type,
            total_length,
        }
    }
}

pub fn normalize_angle(angle: Angle) -> Angle {
    mod2pi(angle as i64)
}

pub fn mod2pi(angle: i64) -> Angle {
    let tau = TAU_MRAD as i64;
    let mut value = angle % tau;
    if value < 0 {
        value += tau;
    }
    value as Angle
}

pub fn scaled_mul(a: i64, b: i64) -> i64 {
    ((a as i128) * (b as i128)).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

pub fn normalized_inputs(start: Pose, end: Pose, rho: Fixed, dtrig: &DTrig) -> (Angle, Angle, Fixed) {
    let dx = scaled_mul(end.x - start.x, SCALE) / rho;
    let dy = scaled_mul(end.y - start.y, SCALE) / rho;
    let d = isqrt_i128((dx as i128) * (dx as i128) + (dy as i128) * (dy as i128));
    let theta = atan2_mrad(dy, dx, dtrig);
    let alpha = mod2pi(start.heading as i64 - theta as i64);
    let beta = mod2pi(end.heading as i64 - theta as i64);
    (alpha, beta, d)
}

pub fn sin_mrad(angle: Angle, dtrig: &DTrig) -> i64 {
    dtrig.sine((angle, 1000)).0 as i64
}

pub fn cos_mrad(angle: Angle, dtrig: &DTrig) -> i64 {
    dtrig.cosine((angle, 1000)).0 as i64
}

pub fn atan2_mrad(y: Fixed, x: Fixed, dtrig: &DTrig) -> Angle {
    if x == 0 && y == 0 {
        return 0;
    }
    if x == 0 {
        return if y > 0 { HALF_PI_MRAD } else { -HALF_PI_MRAD };
    }
    if y == 0 {
        return if x > 0 { 0 } else { PI_MRAD };
    }

    let (num, den) = fit_i32_ratio(y, x);
    let mut angle = dtrig.arctangent((num, den)).0;
    if x < 0 {
        if y >= 0 {
            angle = angle.saturating_add(PI_MRAD);
        } else {
            angle = angle.saturating_sub(PI_MRAD);
        }
    }
    angle
}

pub fn fit_i32_ratio(numer: i64, denom: i64) -> (i32, i32) {
    let max_value = numer.abs().max(denom.abs());
    if max_value == 0 {
        return (0, 1);
    }
    if max_value <= i32::MAX as i64 {
        return (numer as i32, denom as i32);
    }
    let scale = max_value / i32::MAX as i64 + 1;
    ((numer / scale) as i32, (denom / scale) as i32)
}

pub fn arccos_mrad(value_scaled: i64, dtrig: &DTrig) -> Option<Angle> {
    if value_scaled < -1000 || value_scaled > 1000 {
        return None;
    }
    Some(dtrig.arccosine((value_scaled as i32, 1000)).0)
}

pub fn isqrt_i128(value: i128) -> i64 {
    if value <= 0 {
        return 0;
    }
    let mut x = value;
    let mut y = (x + 1) >> 1;
    while y < x {
        x = y;
        y = (x + value / x) >> 1;
    }
    x as i64
}

pub fn dubins_lsl(alpha: Angle, beta: Angle, d: Fixed, dtrig: &DTrig) -> Option<Candidate> {
    let sin_a = sin_mrad(alpha, dtrig);
    let sin_b = sin_mrad(beta, dtrig);
    let cos_a = cos_mrad(alpha, dtrig);
    let cos_b = cos_mrad(beta, dtrig);
    let cos_ab = cos_mrad(mod2pi(alpha as i64 - beta as i64), dtrig);

    let tmp0 = d + sin_a - sin_b;
    let p2 = 2 * SCALE * SCALE
        + scaled_mul(d, d)
        - 2 * cos_ab * SCALE
        + 2 * d * (sin_a - sin_b);
    if p2 < 0 {
        return None;
    }
    let p = isqrt_i128(p2 as i128);
    let tmp1 = atan2_mrad(cos_b - cos_a, tmp0, dtrig);
    let t = mod2pi(-(alpha as i64) + tmp1 as i64);
    let q = mod2pi(beta as i64 - tmp1 as i64);

    Some(Candidate {
        path_type: PathType::LSL,
        t: t as i64,
        p,
        q: q as i64,
    })
}

pub fn dubins_rsr(alpha: Angle, beta: Angle, d: Fixed, dtrig: &DTrig) -> Option<Candidate> {
    let sin_a = sin_mrad(alpha, dtrig);
    let sin_b = sin_mrad(beta, dtrig);
    let cos_a = cos_mrad(alpha, dtrig);
    let cos_b = cos_mrad(beta, dtrig);
    let cos_ab = cos_mrad(mod2pi(alpha as i64 - beta as i64), dtrig);

    let tmp0 = d - sin_a + sin_b;
    let p2 = 2 * SCALE * SCALE
        + scaled_mul(d, d)
        - 2 * cos_ab * SCALE
        + 2 * d * (sin_b - sin_a);
    if p2 < 0 {
        return None;
    }
    let p = isqrt_i128(p2 as i128);
    let tmp1 = atan2_mrad(cos_a - cos_b, tmp0, dtrig);
    let t = mod2pi(alpha as i64 - tmp1 as i64);
    let q = mod2pi(-(beta as i64) + tmp1 as i64);

    Some(Candidate {
        path_type: PathType::RSR,
        t: t as i64,
        p,
        q: q as i64,
    })
}

pub fn dubins_lsr(alpha: Angle, beta: Angle, d: Fixed, dtrig: &DTrig) -> Option<Candidate> {
    let sin_a = sin_mrad(alpha, dtrig);
    let sin_b = sin_mrad(beta, dtrig);
    let cos_a = cos_mrad(alpha, dtrig);
    let cos_b = cos_mrad(beta, dtrig);
    let cos_ab = cos_mrad(mod2pi(alpha as i64 - beta as i64), dtrig);

    let p2 = -2 * SCALE * SCALE
        + scaled_mul(d, d)
        + 2 * cos_ab * SCALE
        + 2 * d * (sin_a + sin_b);
    if p2 < 0 {
        return None;
    }
    let p = isqrt_i128(p2 as i128);
    let tmp2 = atan2_mrad(-cos_a - cos_b, d + sin_a + sin_b, dtrig);
    let tmp3 = atan2_mrad(-2 * SCALE, p, dtrig);
    let t = mod2pi(-(alpha as i64) + (tmp2 as i64 - tmp3 as i64));
    let q = mod2pi(-(beta as i64) + (tmp2 as i64 - tmp3 as i64));

    Some(Candidate {
        path_type: PathType::LSR,
        t: t as i64,
        p,
        q: q as i64,
    })
}

pub fn dubins_rsl(alpha: Angle, beta: Angle, d: Fixed, dtrig: &DTrig) -> Option<Candidate> {
    let sin_a = sin_mrad(alpha, dtrig);
    let sin_b = sin_mrad(beta, dtrig);
    let cos_a = cos_mrad(alpha, dtrig);
    let cos_b = cos_mrad(beta, dtrig);
    let cos_ab = cos_mrad(mod2pi(alpha as i64 - beta as i64), dtrig);

    let p2 = -2 * SCALE * SCALE
        + scaled_mul(d, d)
        + 2 * cos_ab * SCALE
        - 2 * d * (sin_a + sin_b);
    if p2 < 0 {
        return None;
    }
    let p = isqrt_i128(p2 as i128);
    let tmp2 = atan2_mrad(cos_a + cos_b, d - sin_a - sin_b, dtrig);
    let tmp3 = atan2_mrad(2 * SCALE, p, dtrig);
    let t = mod2pi(alpha as i64 - (tmp2 as i64 - tmp3 as i64));
    let q = mod2pi(beta as i64 - (tmp2 as i64 - tmp3 as i64));

    Some(Candidate {
        path_type: PathType::RSL,
        t: t as i64,
        p,
        q: q as i64,
    })
}

pub fn dubins_rlr(alpha: Angle, beta: Angle, d: Fixed, dtrig: &DTrig) -> Option<Candidate> {
    let sin_a = sin_mrad(alpha, dtrig);
    let sin_b = sin_mrad(beta, dtrig);
    let cos_a = cos_mrad(alpha, dtrig);
    let cos_b = cos_mrad(beta, dtrig);
    let cos_ab = cos_mrad(mod2pi(alpha as i64 - beta as i64), dtrig);

    let numerator = 6 * SCALE * SCALE
        - scaled_mul(d, d)
        + 2 * cos_ab * SCALE
        + 2 * d * (sin_a - sin_b);
    let tmp0 = numerator / 8000;
    let acos = arccos_mrad(tmp0, dtrig)?;
    let p = mod2pi(TAU_MRAD as i64 - acos as i64) as i64;
    let tmp1 = atan2_mrad(cos_a - cos_b, d - sin_a + sin_b, dtrig);
    let t = mod2pi(alpha as i64 - tmp1 as i64 + p / 2) as i64;
    let q = mod2pi(alpha as i64 - beta as i64 - t + p) as i64;

    Some(Candidate {
        path_type: PathType::RLR,
        t,
        p,
        q,
    })
}

pub fn dubins_lrl(alpha: Angle, beta: Angle, d: Fixed, dtrig: &DTrig) -> Option<Candidate> {
    let sin_a = sin_mrad(alpha, dtrig);
    let sin_b = sin_mrad(beta, dtrig);
    let cos_a = cos_mrad(alpha, dtrig);
    let cos_b = cos_mrad(beta, dtrig);
    let cos_ab = cos_mrad(mod2pi(alpha as i64 - beta as i64), dtrig);

    let numerator = 6 * SCALE * SCALE
        - scaled_mul(d, d)
        + 2 * cos_ab * SCALE
        + 2 * d * (sin_b - sin_a);
    let tmp0 = numerator / 8000;
    let acos = arccos_mrad(tmp0, dtrig)?;
    let p = mod2pi(TAU_MRAD as i64 - acos as i64) as i64;
    let tmp1 = atan2_mrad(cos_a - cos_b, d + sin_a - sin_b, dtrig);
    let t = mod2pi(-(alpha as i64) - tmp1 as i64 + p / 2) as i64;
    let q = mod2pi(beta as i64 - alpha as i64 - t + p) as i64;

    Some(Candidate {
        path_type: PathType::LRL,
        t,
        p,
        q,
    })
}

pub fn propagate_segment(
    pose: Pose,
    segment: DubinsSegment,
    distance: Fixed,
    rho: Fixed,
    dtrig: &DTrig,
) -> Pose {
    match segment.kind {
        SegmentType::Straight => {
            let cos_h = cos_mrad(pose.heading, dtrig);
            let sin_h = sin_mrad(pose.heading, dtrig);
            let dx = scaled_mul(cos_h, distance) / SCALE;
            let dy = scaled_mul(sin_h, distance) / SCALE;
            Pose::new_fixed(pose.x + dx, pose.y + dy, pose.heading)
        }
        SegmentType::Left => {
            let angle = scaled_mul(distance, SCALE) / rho;
            let new_heading = mod2pi(pose.heading as i64 + angle as i64);
            let sin0 = sin_mrad(pose.heading, dtrig);
            let cos0 = cos_mrad(pose.heading, dtrig);
            let sin1 = sin_mrad(new_heading, dtrig);
            let cos1 = cos_mrad(new_heading, dtrig);
            let dx = scaled_mul(rho, sin1 - sin0) / SCALE;
            let dy = scaled_mul(rho, cos0 - cos1) / SCALE;
            Pose::new_fixed(pose.x + dx, pose.y + dy, new_heading)
        }
        SegmentType::Right => {
            let angle = scaled_mul(distance, SCALE) / rho;
            let new_heading = mod2pi(pose.heading as i64 - angle as i64);
            let sin0 = sin_mrad(pose.heading, dtrig);
            let cos0 = cos_mrad(pose.heading, dtrig);
            let sin1 = sin_mrad(new_heading, dtrig);
            let cos1 = cos_mrad(new_heading, dtrig);
            let dx = scaled_mul(rho, sin0 - sin1) / SCALE;
            let dy = scaled_mul(rho, cos1 - cos0) / SCALE;
            Pose::new_fixed(pose.x + dx, pose.y + dy, new_heading)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_path() {
        let ctx = DubinsContext::new();
        let start = Pose::new_units(0, 0, 0);
        let end = Pose::new_units(10, 0, 0);
        let rho = 1 * SCALE;
        let path = ctx.shortest_path(start, end, rho).unwrap();
        assert_eq!(path.total_length, 10 * SCALE);
        let end_pose = ctx.sample(&path, path.total_length);
        assert!((end_pose.x - end.x).abs() <= 1);
        assert!((end_pose.y - end.y).abs() <= 1);
        assert_eq!(end_pose.heading, end.heading);
    }

    #[test]
    fn offset_turn_path_reaches_goal() {
        let ctx = DubinsContext::new();
        let start = Pose::new_units(0, 0, 0);
        let end = Pose::new_units(6, 6, HALF_PI_MRAD);
        let rho = 2 * SCALE;
        let path = ctx.shortest_path(start, end, rho).unwrap();
        let end_pose = ctx.sample(&path, path.total_length);
        assert!((end_pose.x - end.x).abs() <= 5);
        assert!((end_pose.y - end.y).abs() <= 5);
        assert_eq!(end_pose.heading, end.heading);
    }
}

pub fn sample_path_range(
    dubins: &DubinsContext,
    path: &DubinsPath,
    step: Fixed,
    start: Fixed,
    end: Fixed,
) -> Vec<FixedVec2> {
    let mut points = Vec::new();
    if step <= 0 || end < start {
        return points;
    }
    let mut distance = start;
    while distance < end {
        let pose = dubins.sample(path, distance);
        points.push(FixedVec2::new(pose.x, pose.y));
        distance = distance.saturating_add(step);
    }
    let pose = dubins.sample(path, end);
    points.push(FixedVec2::new(pose.x, pose.y));
    points
}

pub fn fixed_from_f32(value: f32) -> Fixed {
    (value * SCALE as f32).round() as Fixed
}

pub fn fixed_to_f32(value: Fixed) -> f32 {
    value as f32 / SCALE as f32
}

pub fn angle_to_f32(angle: Angle) -> f32 {
    angle as f32 / 1000.0
}

pub fn unit_vector_mrad(angle: Angle, dtrig: &DTrig) -> (Fixed, Fixed) {
    let sin = dtrig.sine((angle, 1000)).0 as i64;
    let cos = dtrig.cosine((angle, 1000)).0 as i64;
    (cos, sin)
}

pub fn fixed_from_angle(center: FixedVec2, radius: Fixed, angle: Angle, dtrig: &DTrig) -> FixedVec2 {
    let (cos, sin) = unit_vector_mrad(angle, dtrig);
    let dx = scaled_mul_i64(radius, cos) / SCALE;
    let dy = scaled_mul_i64(radius, sin) / SCALE;
    FixedVec2::new(center.x + dx, center.y + dy)
}

pub fn scaled_mul_i64(a: i64, b: i64) -> i64 {
    ((a as i128) * (b as i128)).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

pub fn mul_div(a: i64, b: i64, denom: i64) -> i64 {
    if denom == 0 {
        return 0;
    }
    ((a as i128) * (b as i128) / (denom as i128)) as i64
}

pub fn angular_delta(speed: Fixed, radius: Fixed, dt_micros: i64, micros_per_sec: i64) -> Angle {
    if radius <= 0 {
        return 0;
    }
    let omega_mrad = mul_div(speed, 1000, radius);
    mul_div(omega_mrad, dt_micros, micros_per_sec) as Angle
}