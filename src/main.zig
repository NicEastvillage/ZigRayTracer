const std = @import("std");
const rl = @import("raylib");
const rlm = @import("raylib-math");

// External types
const Timer = std.time.Timer;
const Thread = std.Thread;
const Vec3 = rl.Vector3;
const Color = rl.Color;
const Ray = rl.Ray;
const RayCollision = rl.RayCollision;
const Rng = std.rand.DefaultPrng;

// External functions
const print = std.debug.print;
const minV = rlm.Vector3Min;
const maxV = rlm.Vector3Max;
const scale = rlm.Vector3Scale;
const add = rlm.Vector3Add;
const sub = rlm.Vector3Subtract;
const mul = rlm.Vector3Multiply;
const dot = rlm.Vector3DotProduct;
const normalize = rlm.Vector3Normalize;
const reflect = rlm.Vector3Reflect;
const magnitude = rlm.Vector3Length;
const min = std.math.min;
const pow = std.math.pow;
const fabs = std.math.fabs;

// Handy values
const zeroes = Vec3{ .x = 0, .y = 0, .z = 0 };
const ones = Vec3{ .x = 1, .y = 1, .z = 1 };

// Rendering settings
//--------------------------------------------------------------------------------------
const screen_width = 1280;
const screen_height = 680;
const samples_per_pixel = 4;
const sphere_count = 130;
const max_depth = 8;
const thread_count = 14;
const sky_color = Vec3{ .x = 0.4, .y = 0.55, .z = 0.92 };
const ground_color = Vec3{ .x = 0.8, .y = 0.6, .z = 0.5 };

threadlocal var rng = Rng.init(12321);

// Materials and Sphere
//--------------------------------------------------------------------------------------
const Lambertian = struct {
    albedo: Vec3,
};

const Metal = struct {
    albedo: Vec3,
    fuzz: f32,
};

const Dialectic = struct {
    albedo: Vec3,
    ir: f32,
};

const Material = union(enum) {
    lambertian: Lambertian,
    metal: Metal,
    dialectic: Dialectic,
};

const Sphere = struct {
    pos: Vec3,
    radius: f32,
    material: Material,
};

// Utility functions
//--------------------------------------------------------------------------------------
fn sqrtV(vec: Vec3) Vec3 {
    return Vec3{ .x = @sqrt(vec.x), .y = @sqrt(vec.y), .z = @sqrt(vec.z) };
}

fn colorToVec3(col: Color) Vec3 {
    return Vec3{ .x = @intToFloat(f32, col.r) / 255.0, .y = @intToFloat(f32, col.g) / 255.0, .z = @intToFloat(f32, col.b) / 255.0 };
}

fn vec3ToColor(vec: Vec3, samples: u32) Color {
    const s = @intToFloat(f32, samples);
    // We take the square root to gamma correct with gamma=2
    const v = minV(maxV(zeroes, scale(sqrtV(scale(vec, 1.0 / s)), 255.0)), Vec3{ .x = 255, .y = 255, .z = 255 });
    return Color{ .r = @floatToInt(u8, v.x), .g = @floatToInt(u8, v.y), .b = @floatToInt(u8, v.z), .a = 255 };
}

fn sign(comptime T: type, num: T) T {
    if (num >= 0) {
        return 1;
    } else {
        return -1;
    }
}

fn random01Vec3() Vec3 {
    return Vec3{ .x = rng.random().float(f32), .y = rng.random().float(f32), .z = rng.random().float(f32) };
}

fn randomVec3InUnitShpere() Vec3 {
    while (true) {
        const v = sub(scale(random01Vec3(), 2), ones);
        if (magnitude(v) < 1.0) return v;
    }
}

fn randomVec3InUnitHemiSphere(normal: Vec3) Vec3 {
    const v = randomVec3InUnitShpere();
    return scale(v, sign(f32, dot(v, normal)));
}

fn refract(vec: Vec3, normal: Vec3, eta_ratio: f32) Vec3 {
    const cos_theta = min(dot(scale(vec, -1.0), normal), 1.0);
    const out_perp = scale(add(vec, scale(normal, cos_theta)), eta_ratio);
    const out_para_neg = scale(normal, @sqrt(1.0 - pow(f32, magnitude(out_perp), 2)));
    return sub(out_perp, out_para_neg);
}

// Ray tracing stuff
//--------------------------------------------------------------------------------------
fn rayColor(ray: Ray, spheres: []const Sphere, depth: u32) Vec3 {
    if (depth == 0) {
        return zeroes;
    }

    // Find closest hit sphere
    var closest = RayCollision{
        .hit = false,
        .distance = std.math.f32_max,
        .position = zeroes,
        .normal = zeroes,
    };
    var material: Material = undefined;
    for (spheres) |sphere| {
        const collision = rl.GetRayCollisionSphere(ray, sphere.pos, sphere.radius);
        if (collision.hit and collision.distance > 0.01 and collision.distance < closest.distance) {
            closest = collision;
            material = sphere.material;
        }
    }
    if (!closest.hit) {
        // We hit nothing, using sky light
        const t = @maximum(0.5 * (ray.direction.y + 1), 0);
        return rlm.Vector3Lerp(ones, sky_color, t);
    }

    // Find attenuation and reflected ray
    var new_ray: Ray = undefined;
    var attenuation: Vec3 = undefined;
    var new_depth = depth - 1;
    switch (material) {
        Material.lambertian => |lambertian| {
            attenuation = lambertian.albedo;

            // For lambertian materials the ray's direction is the normal plus some randomness
            const direction = normalize(add(closest.normal, normalize(randomVec3InUnitShpere())));
            new_ray = Ray{ .position = closest.position, .direction = direction };
        },
        Material.metal => |metal| {
            attenuation = metal.albedo;

            // Metals intuitively reflect the ray in the normal
            const direction = normalize(add(reflect(ray.direction, closest.normal), scale(randomVec3InUnitShpere(), metal.fuzz)));
            new_ray = Ray{ .position = closest.position, .direction = direction };
        },
        Material.dialectic => |dialectic| {
            attenuation = dialectic.albedo;

            // Technically glass both reflect and refract. We always refract if possible
            const refract_ratio = if (dot(ray.direction, closest.normal) > 0) dialectic.ir else 1.0 / dialectic.ir;
            const cos_theta = min(dot(scale(ray.direction, -1.0), closest.normal), 1.0);
            const sin_theta = @sqrt(1.0 - pow(f32, cos_theta, 2));
            const cannot_refract = refract_ratio * sin_theta > 1.0;

            // The ray must reflect near the edge. Schlick's approximation
            const r0 = pow(f32, (1.0 - refract_ratio) / (1.0 + refract_ratio), 2);
            const reflectance = r0 + (1.0 - r0) * pow(f32, 1.0 - cos_theta, 5);
            const must_reflect = rng.random().float(f32) < reflectance;

            const direction = if (cannot_refract or must_reflect)
                reflect(ray.direction, closest.normal)
            else
                refract(ray.direction, closest.normal, refract_ratio);

            new_ray = Ray{ .position = closest.position, .direction = direction };
            // We can easily get away with a low max depth, but glass will look weird. Hence glass will not decrease depth
            new_depth = depth;
        },
    }

    return mul(attenuation, rayColor(new_ray, spheres, depth - 1));
}

/// Renders the scanlines from v0 to v1
fn renderSection(img_buffer: []Color, v0: i32, v1: i32, spheres: []const Sphere, cam: *const rl.Camera3D) void {
    var u: i32 = 0;
    while (u < screen_width) : (u += 1) {
        var v: i32 = v0;
        while (v < v1) : (v += 1) {
            const uf = @intToFloat(f32, u);
            const vf = @intToFloat(f32, screen_height - v - 1);
            var col = zeroes;
            // Make multiple samples per pixel with slight offset to achieve anti-aliasing
            var i: i32 = 0;
            while (i < samples_per_pixel) : (i += 1) {
                const du = rng.random().float(f32);
                const dv = rng.random().float(f32);
                const ray = rl.GetMouseRay(rl.Vector2{ .x = uf + du, .y = vf + dv }, cam.*);
                col = add(col, rayColor(ray, spheres, max_depth));
            }
            const c = vec3ToColor(col, samples_per_pixel);
            img_buffer[@intCast(usize, u + v * screen_width)] = c;
        }
    }
}

// Setup and main
//--------------------------------------------------------------------------------------
fn createSpheres(comptime count: usize) [count + 1]Sphere {
    var spheres: [count + 1]Sphere = undefined;

    // The ground is also a huge sphere
    spheres[0] = Sphere{ .pos = Vec3{ .x = 0, .y = -10000, .z = 0 }, .radius = 10000, .material = Material{ .lambertian = Lambertian{ .albedo = ground_color } } };

    var i: usize = 1;
    while (i < count + 1) : (i += 1) {
        var x: f32 = undefined;
        var y: f32 = undefined;
        var z: f32 = undefined;

        // Avoid overlap with existing circles
        var overlap = true;
        var tries: i32 = 1000;
        while (overlap and tries >= 0) : (tries -= 1) {
            x = 40 - rng.random().float(f32) * 80;
            y = 1.0 + rng.random().float(f32) * rng.random().float(f32) * 2;
            z = 25 - rng.random().float(f32) * 95;

            overlap = false;
            var j: usize = 1;
            while (j < i) : (j += 1) {
                const other = &spheres[j];
                if (@sqrt(pow(f32, x - other.pos.x, 2) + pow(f32, z - other.pos.z, 2)) < other.radius + y) {
                    overlap = true;
                    break;
                }
            }
        }

        const pos = Vec3{ .x = x, .y = y, .z = z };

        // Choose material
        var material: Material = undefined;
        const mat_rng = rng.random().float(f32);
        if (mat_rng < 0.6) {
            const albedo = random01Vec3();
            material = Material{ .lambertian = Lambertian{ .albedo = albedo } };
        } else if (mat_rng < 0.85) {
            const albedo = random01Vec3();
            const fuzz = rng.random().float(f32) * rng.random().float(f32);
            material = Material{ .metal = Metal{ .albedo = albedo, .fuzz = fuzz } };
        } else {
            const albedo = sub(ones, scale(random01Vec3(), 0.4));
            const ir = 1.0 + rng.random().float(f32) * 1.5;
            material = Material{ .dialectic = Dialectic{ .albedo = albedo, .ir = ir } };
        }

        spheres[i] = Sphere{ .pos = pos, .radius = y, .material = material };
    }

    return spheres;
}

pub fn main() anyerror!void {
    // Initialization
    //--------------------------------------------------------------------------------------
    rl.InitWindow(screen_width, screen_height, "ZigRayTracer");
    defer rl.CloseWindow();

    rl.SetTargetFPS(60);

    const cam = rl.Camera3D{
        .position = Vec3{ .x = 0, .y = 8.0, .z = 35 },
        .target = Vec3{ .x = 0, .y = 2, .z = 0 },
        .up = Vec3{ .x = 0, .y = 1, .z = 0 },
        .fovy = 45.0,
        .projection = rl.CameraProjection.CAMERA_PERSPECTIVE,
    };

    const spheres = createSpheres(sphere_count);

    // Render
    //--------------------------------------------------------------------------------------
    var img_buffer: [screen_width * screen_height]Color = undefined;
    const sectionHeight = screen_height / thread_count;
    var threads: [thread_count]Thread = undefined;
    const renderTimer = try Timer.start();

    // Start threads - each is tasked with a section of the buffer
    var tid: i32 = 0;
    while (tid < thread_count) : (tid += 1) {
        const v0 = tid * sectionHeight;
        const v1 = if (tid == thread_count - 1) screen_height else tid * sectionHeight + sectionHeight;
        print("Thread {} will render v={}..{}\n", .{ tid, v0, v1 });
        threads[@intCast(usize, tid)] = try Thread.spawn(.{}, renderSection, .{ img_buffer[0..], v0, v1, spheres[0..], &cam });
    }

    // Wait for rendering to finish
    for (threads) |thread| {
        thread.join();
    }

    // Draw buffer on texture
    var texture = rl.LoadRenderTexture(screen_width, screen_height);
    texture.Begin();
    var u: i32 = 0;
    while (u < screen_width) : (u += 1) {
        var v: i32 = 0;
        while (v < screen_height) : (v += 1) {
            const c = img_buffer[@intCast(usize, u + v * screen_width)];
            rl.DrawPixel(u, v, c);
        }
    }
    texture.End();

    print("Finished rendering in {d:.3} seconds\n", .{0.000000001 * @intToFloat(f32, renderTimer.read())});

    // Main loop
    //--------------------------------------------------------------------------------------
    while (!rl.WindowShouldClose()) {
        // Draw
        rl.BeginDrawing();
        rl.ClearBackground(rl.BLACK);
        rl.DrawTexture(texture.texture, 0, 0, rl.WHITE);
        rl.EndDrawing();
    }
}
