const std = @import("std");
const Builder = std.build.Builder;
const raylib = @import("raylib-zig/lib.zig").Pkg("raylib-zig"); //call .Pkg() with the folder raylib-zig is in relative to project build.zig

pub fn build(b: *Builder) void {
    const mode = b.standardReleaseOptions();
    const target = b.standardTargetOptions(.{});

    const system_lib = b.option(bool, "system-raylib", "link to preinstalled raylib libraries") orelse false;

    const exe = b.addExecutable("ZigRayTracer", "src/main.zig");
    exe.setBuildMode(mode);
    exe.setTarget(target);

    raylib.link(exe, system_lib);
    raylib.addAsPackage("raylib", exe);
    raylib.math.addAsPackage("raylib-math", exe);

    const run_cmd = exe.run();
    const run_step = b.step("run", "run ZigRayTracer");
    run_step.dependOn(&run_cmd.step);

    exe.install();
}
