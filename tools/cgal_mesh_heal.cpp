#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/shape_predicates.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/boost/graph/helpers.h>

#include <boost/graph/graph_traits.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
using GraphTraits = boost::graph_traits<Mesh>;
using Face = GraphTraits::face_descriptor;
using Halfedge = GraphTraits::halfedge_descriptor;

struct Options {
  std::string input_mesh;
  std::string output_mesh;
  std::string summary_json;
  std::string filled_patch_mesh;
  std::string hole_report_jsonl;
  bool repair_degenerate_faces = true;
  bool repair_almost_degenerate_faces = false;
  bool duplicate_nonmanifold_vertices = true;
  bool stitch_borders = true;
  bool fill_holes = true;
  int max_hole_edges = 48;
  double max_hole_diameter = 0.0;
  double component_area_threshold = 0.0;
  bool repair_self_intersections = false;
  bool detect_self_intersections = true;
};

struct MeshStats {
  std::size_t vertices = 0;
  std::size_t faces = 0;
  std::size_t edges = 0;
  std::size_t boundary_edges = 0;
  std::size_t boundary_cycles = 0;
  std::size_t connected_components = 0;
  std::size_t nonmanifold_vertices = 0;
  std::size_t degenerate_faces = 0;
  long long self_intersection_pairs = -1;
  double surface_area = 0.0;
  double bbox_diagonal = 0.0;
};

struct HoleMeasure {
  std::size_t edges = 0;
  double bbox_diagonal = 0.0;
};

struct Operations {
  bool degenerate_faces_repair_ok = true;
  bool almost_degenerate_faces_repair_ok = true;
  std::size_t duplicated_nonmanifold_vertices = 0;
  std::size_t stitched_halfedge_pairs = 0;
  std::size_t selected_holes = 0;
  std::size_t filled_holes = 0;
  std::size_t hole_patch_faces = 0;
  std::size_t hole_patch_vertices = 0;
  std::size_t removed_tiny_components = 0;
  bool self_intersection_repair_ok = true;
  std::string self_intersection_repair_error;
};

struct HoleRecord {
  std::size_t id = 0;
  std::size_t boundary_edges = 0;
  double bbox_diagonal = 0.0;
  bool edge_gate = true;
  bool diameter_gate = true;
  bool selected = false;
  bool filled = false;
  std::size_t patch_faces = 0;
  std::size_t patch_vertices = 0;
};

struct HoleReport {
  std::vector<HoleRecord> records;
  std::size_t skipped_by_edge_limit = 0;
  std::size_t skipped_by_diameter_limit = 0;
};

static void print_usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " INPUT_MESH OUTPUT_MESH SUMMARY_JSON [options]\n\n"
      << "Options:\n"
      << "  --no-repair-degenerate-faces\n"
      << "  --repair-almost-degenerate-faces\n"
      << "  --no-duplicate-nonmanifold-vertices\n"
      << "  --no-stitch-borders\n"
      << "  --no-fill-holes\n"
      << "  --max-hole-edges N\n"
      << "  --max-hole-diameter D\n"
      << "  --component-area-threshold A\n"
      << "  --repair-self-intersections\n"
      << "  --no-detect-self-intersections\n"
      << "  --filled-patch-mesh PATH\n"
      << "  --hole-report-jsonl PATH\n";
}

static double parse_double(const std::string& value, const std::string& flag) {
  char* end = nullptr;
  const double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0') {
    throw std::runtime_error("Invalid float for " + flag + ": " + value);
  }
  return parsed;
}

static int parse_int(const std::string& value, const std::string& flag) {
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') {
    throw std::runtime_error("Invalid integer for " + flag + ": " + value);
  }
  return static_cast<int>(parsed);
}

static Options parse_args(int argc, char** argv) {
  if (argc == 2) {
    const std::string only_arg = argv[1];
    if (only_arg == "--help" || only_arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
  }
  if (argc < 4) {
    print_usage(argv[0]);
    throw std::runtime_error("Missing required paths.");
  }
  Options opts;
  opts.input_mesh = argv[1];
  opts.output_mesh = argv[2];
  opts.summary_json = argv[3];

  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
      }
      return argv[++i];
    };

    if (arg == "--no-repair-degenerate-faces") {
      opts.repair_degenerate_faces = false;
    } else if (arg == "--repair-almost-degenerate-faces") {
      opts.repair_almost_degenerate_faces = true;
    } else if (arg == "--no-duplicate-nonmanifold-vertices") {
      opts.duplicate_nonmanifold_vertices = false;
    } else if (arg == "--no-stitch-borders") {
      opts.stitch_borders = false;
    } else if (arg == "--no-fill-holes") {
      opts.fill_holes = false;
    } else if (arg == "--max-hole-edges") {
      opts.max_hole_edges = parse_int(require_value(arg), arg);
    } else if (arg == "--max-hole-diameter") {
      opts.max_hole_diameter = parse_double(require_value(arg), arg);
    } else if (arg == "--component-area-threshold") {
      opts.component_area_threshold = parse_double(require_value(arg), arg);
    } else if (arg == "--repair-self-intersections") {
      opts.repair_self_intersections = true;
    } else if (arg == "--no-detect-self-intersections") {
      opts.detect_self_intersections = false;
    } else if (arg == "--filled-patch-mesh") {
      opts.filled_patch_mesh = require_value(arg);
    } else if (arg == "--hole-report-jsonl") {
      opts.hole_report_jsonl = require_value(arg);
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  return opts;
}

static std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c; break;
    }
  }
  return out.str();
}

static std::size_t count_boundary_edges(const Mesh& mesh) {
  std::size_t count = 0;
  for (Halfedge h : halfedges(mesh)) {
    if (is_border(h, mesh)) {
      ++count;
    }
  }
  return count;
}

static std::size_t count_boundary_cycles(const Mesh& mesh) {
  std::vector<Halfedge> cycles;
  PMP::extract_boundary_cycles(mesh, std::back_inserter(cycles));
  return cycles.size();
}

static std::size_t count_connected_components(Mesh& mesh) {
  if (num_faces(mesh) == 0) {
    return 0;
  }
  auto prop = mesh.add_property_map<Face, std::size_t>("f:omega_tmp_cc", 0);
  const std::size_t count = PMP::connected_components(mesh, prop.first);
  mesh.remove_property_map(prop.first);
  return count;
}

static std::size_t count_nonmanifold_vertices(const Mesh& mesh) {
  std::vector<Halfedge> cones;
  PMP::non_manifold_vertices(mesh, std::back_inserter(cones));
  return cones.size();
}

static std::size_t count_degenerate_faces(const Mesh& mesh) {
  std::vector<Face> degenerate;
  PMP::degenerate_faces(mesh, std::back_inserter(degenerate));
  return degenerate.size();
}

static long long count_self_intersection_pairs(const Mesh& mesh) {
  std::vector<std::pair<Face, Face>> pairs;
  PMP::self_intersections<CGAL::Sequential_tag>(faces(mesh), mesh, std::back_inserter(pairs));
  return static_cast<long long>(pairs.size());
}

static double compute_bbox_diagonal(const Mesh& mesh) {
  if (num_vertices(mesh) == 0) {
    return 0.0;
  }
  double xmin = std::numeric_limits<double>::infinity();
  double ymin = std::numeric_limits<double>::infinity();
  double zmin = std::numeric_limits<double>::infinity();
  double xmax = -std::numeric_limits<double>::infinity();
  double ymax = -std::numeric_limits<double>::infinity();
  double zmax = -std::numeric_limits<double>::infinity();
  for (auto v : vertices(mesh)) {
    const Point& p = mesh.point(v);
    xmin = std::min(xmin, p.x());
    ymin = std::min(ymin, p.y());
    zmin = std::min(zmin, p.z());
    xmax = std::max(xmax, p.x());
    ymax = std::max(ymax, p.y());
    zmax = std::max(zmax, p.z());
  }
  const double dx = xmax - xmin;
  const double dy = ymax - ymin;
  const double dz = zmax - zmin;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static MeshStats collect_stats(Mesh& mesh, bool detect_self_intersections) {
  MeshStats stats;
  stats.vertices = num_vertices(mesh);
  stats.faces = num_faces(mesh);
  stats.edges = num_edges(mesh);
  stats.boundary_edges = count_boundary_edges(mesh);
  stats.boundary_cycles = count_boundary_cycles(mesh);
  stats.connected_components = count_connected_components(mesh);
  stats.nonmanifold_vertices = count_nonmanifold_vertices(mesh);
  stats.degenerate_faces = count_degenerate_faces(mesh);
  stats.surface_area = CGAL::to_double(PMP::area(mesh));
  stats.bbox_diagonal = compute_bbox_diagonal(mesh);
  if (detect_self_intersections) {
    stats.self_intersection_pairs = count_self_intersection_pairs(mesh);
  }
  return stats;
}

static HoleMeasure measure_hole_cycle(const Mesh& mesh, Halfedge start) {
  HoleMeasure measure;
  if (start == GraphTraits::null_halfedge()) {
    return measure;
  }
  double xmin = std::numeric_limits<double>::infinity();
  double ymin = std::numeric_limits<double>::infinity();
  double zmin = std::numeric_limits<double>::infinity();
  double xmax = -std::numeric_limits<double>::infinity();
  double ymax = -std::numeric_limits<double>::infinity();
  double zmax = -std::numeric_limits<double>::infinity();

  Halfedge h = start;
  const std::size_t guard = num_halfedges(mesh) + 1;
  do {
    const Point& p = mesh.point(target(h, mesh));
    xmin = std::min(xmin, p.x());
    ymin = std::min(ymin, p.y());
    zmin = std::min(zmin, p.z());
    xmax = std::max(xmax, p.x());
    ymax = std::max(ymax, p.y());
    zmax = std::max(zmax, p.z());
    ++measure.edges;
    h = next(h, mesh);
  } while (h != start && measure.edges < guard);

  const double dx = xmax - xmin;
  const double dy = ymax - ymin;
  const double dz = zmax - zmin;
  measure.bbox_diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
  return measure;
}

static void append_patch_faces(const Mesh& mesh, const std::vector<Face>& patch_faces, Mesh& patch_mesh) {
  for (Face face_descriptor : patch_faces) {
    std::vector<Mesh::Vertex_index> copied_vertices;
    for (Halfedge h : halfedges_around_face(halfedge(face_descriptor, mesh), mesh)) {
      copied_vertices.push_back(patch_mesh.add_vertex(mesh.point(target(h, mesh))));
    }
    if (copied_vertices.size() >= 3) {
      patch_mesh.add_face(copied_vertices);
    }
  }
}

static void fill_selected_holes(Mesh& mesh, const Options& opts, Operations& ops, HoleReport& report, Mesh& patch_mesh) {
  std::vector<Halfedge> cycles;
  PMP::extract_boundary_cycles(mesh, std::back_inserter(cycles));
  for (Halfedge h : cycles) {
    HoleRecord record;
    record.id = report.records.size();
    const HoleMeasure measure = measure_hole_cycle(mesh, h);
    record.boundary_edges = measure.edges;
    record.bbox_diagonal = measure.bbox_diagonal;
    const bool edge_ok = opts.max_hole_edges <= 0 || static_cast<int>(measure.edges) <= opts.max_hole_edges;
    const bool diameter_ok = opts.max_hole_diameter <= 0.0 || measure.bbox_diagonal <= opts.max_hole_diameter;
    record.edge_gate = edge_ok;
    record.diameter_gate = diameter_ok;
    if (!edge_ok || !diameter_ok) {
      if (!edge_ok) {
        ++report.skipped_by_edge_limit;
      }
      if (!diameter_ok) {
        ++report.skipped_by_diameter_limit;
      }
      report.records.push_back(record);
      continue;
    }
    ++ops.selected_holes;
    record.selected = true;
    std::vector<Face> patch_faces;
    std::vector<GraphTraits::vertex_descriptor> patch_vertices;
    PMP::triangulate_and_refine_hole(
        mesh,
        h,
        CGAL::parameters::face_output_iterator(std::back_inserter(patch_faces))
            .vertex_output_iterator(std::back_inserter(patch_vertices)));
    if (!patch_faces.empty()) {
      ++ops.filled_holes;
      ops.hole_patch_faces += patch_faces.size();
      ops.hole_patch_vertices += patch_vertices.size();
      record.filled = true;
      record.patch_faces = patch_faces.size();
      record.patch_vertices = patch_vertices.size();
      append_patch_faces(mesh, patch_faces, patch_mesh);
    }
    report.records.push_back(record);
  }
}

static std::vector<double> quantiles(std::vector<double> values) {
  if (values.empty()) {
    return std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  std::sort(values.begin(), values.end());
  std::vector<double> out;
  for (double q : {0.0, 0.25, 0.5, 0.75, 0.9, 0.99}) {
    const double position = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = std::min<std::size_t>(lo + 1, values.size() - 1);
    const double t = position - static_cast<double>(lo);
    out.push_back(values[lo] * (1.0 - t) + values[hi] * t);
  }
  return out;
}

static void write_json_number_array(std::ostream& out, const std::vector<double>& values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) {
      out << ", ";
    }
    out << std::setprecision(17) << values[i];
  }
  out << "]";
}

static void write_hole_report_jsonl(const Options& opts, const HoleReport& report) {
  if (opts.hole_report_jsonl.empty()) {
    return;
  }
  std::ofstream out(opts.hole_report_jsonl);
  if (!out) {
    throw std::runtime_error("Could not write hole report jsonl: " + opts.hole_report_jsonl);
  }
  for (const HoleRecord& record : report.records) {
    out << "{"
        << "\"id\":" << record.id << ","
        << "\"boundaryEdges\":" << record.boundary_edges << ","
        << "\"bboxDiagonal\":" << std::setprecision(17) << record.bbox_diagonal << ","
        << "\"edgeGate\":" << (record.edge_gate ? "true" : "false") << ","
        << "\"diameterGate\":" << (record.diameter_gate ? "true" : "false") << ","
        << "\"selected\":" << (record.selected ? "true" : "false") << ","
        << "\"filled\":" << (record.filled ? "true" : "false") << ","
        << "\"patchFaces\":" << record.patch_faces << ","
        << "\"patchVertices\":" << record.patch_vertices
        << "}\n";
  }
}

static void write_stats(std::ostream& out, const MeshStats& stats, const std::string& indent) {
  out << indent << "\"vertices\": " << stats.vertices << ",\n";
  out << indent << "\"faces\": " << stats.faces << ",\n";
  out << indent << "\"edges\": " << stats.edges << ",\n";
  out << indent << "\"boundaryEdges\": " << stats.boundary_edges << ",\n";
  out << indent << "\"boundaryCycles\": " << stats.boundary_cycles << ",\n";
  out << indent << "\"connectedComponents\": " << stats.connected_components << ",\n";
  out << indent << "\"nonmanifoldVertices\": " << stats.nonmanifold_vertices << ",\n";
  out << indent << "\"degenerateFaces\": " << stats.degenerate_faces << ",\n";
  out << indent << "\"selfIntersectionPairs\": " << stats.self_intersection_pairs << ",\n";
  out << indent << "\"surfaceArea\": " << std::setprecision(17) << stats.surface_area << ",\n";
  out << indent << "\"bboxDiagonal\": " << std::setprecision(17) << stats.bbox_diagonal << "\n";
}

static void write_summary(
    const Options& opts,
    const MeshStats& before,
    const MeshStats& after,
    const Operations& ops,
    const HoleReport& hole_report) {
  std::ofstream out(opts.summary_json);
  if (!out) {
    throw std::runtime_error("Could not write summary: " + opts.summary_json);
  }
  out << "{\n";
  out << "  \"stageName\": \"omega_local.remesh.cgal_mesh_heal\",\n";
  out << "  \"backend\": \"cgal_polygon_mesh_processing\",\n";
  out << "  \"inputMesh\": \"" << json_escape(opts.input_mesh) << "\",\n";
  out << "  \"outputMesh\": \"" << json_escape(opts.output_mesh) << "\",\n";
  out << "  \"config\": {\n";
  out << "    \"repairDegenerateFaces\": " << (opts.repair_degenerate_faces ? "true" : "false") << ",\n";
  out << "    \"repairAlmostDegenerateFaces\": " << (opts.repair_almost_degenerate_faces ? "true" : "false") << ",\n";
  out << "    \"duplicateNonmanifoldVertices\": " << (opts.duplicate_nonmanifold_vertices ? "true" : "false") << ",\n";
  out << "    \"stitchBorders\": " << (opts.stitch_borders ? "true" : "false") << ",\n";
  out << "    \"fillHoles\": " << (opts.fill_holes ? "true" : "false") << ",\n";
  out << "    \"maxHoleEdges\": " << opts.max_hole_edges << ",\n";
  out << "    \"maxHoleDiameter\": " << std::setprecision(17) << opts.max_hole_diameter << ",\n";
  out << "    \"componentAreaThreshold\": " << std::setprecision(17) << opts.component_area_threshold << ",\n";
  out << "    \"repairSelfIntersections\": " << (opts.repair_self_intersections ? "true" : "false") << ",\n";
  out << "    \"detectSelfIntersections\": " << (opts.detect_self_intersections ? "true" : "false") << ",\n";
  out << "    \"filledPatchMesh\": \"" << json_escape(opts.filled_patch_mesh) << "\",\n";
  out << "    \"holeReportJsonl\": \"" << json_escape(opts.hole_report_jsonl) << "\"\n";
  out << "  },\n";
  out << "  \"operations\": {\n";
  out << "    \"degenerateFacesRepairOk\": " << (ops.degenerate_faces_repair_ok ? "true" : "false") << ",\n";
  out << "    \"almostDegenerateFacesRepairOk\": " << (ops.almost_degenerate_faces_repair_ok ? "true" : "false") << ",\n";
  out << "    \"duplicatedNonmanifoldVertices\": " << ops.duplicated_nonmanifold_vertices << ",\n";
  out << "    \"stitchedHalfedgePairs\": " << ops.stitched_halfedge_pairs << ",\n";
  out << "    \"selectedHoles\": " << ops.selected_holes << ",\n";
  out << "    \"filledHoles\": " << ops.filled_holes << ",\n";
  out << "    \"holePatchFaces\": " << ops.hole_patch_faces << ",\n";
  out << "    \"holePatchVertices\": " << ops.hole_patch_vertices << ",\n";
  out << "    \"removedTinyComponents\": " << ops.removed_tiny_components << ",\n";
  out << "    \"selfIntersectionRepairOk\": " << (ops.self_intersection_repair_ok ? "true" : "false") << ",\n";
  out << "    \"selfIntersectionRepairError\": \"" << json_escape(ops.self_intersection_repair_error) << "\"\n";
  out << "  },\n";
  std::vector<double> hole_edges;
  std::vector<double> hole_diameters;
  hole_edges.reserve(hole_report.records.size());
  hole_diameters.reserve(hole_report.records.size());
  for (const HoleRecord& record : hole_report.records) {
    hole_edges.push_back(static_cast<double>(record.boundary_edges));
    hole_diameters.push_back(record.bbox_diagonal);
  }
  out << "  \"holeReport\": {\n";
  out << "    \"candidateHolesBefore\": " << hole_report.records.size() << ",\n";
  out << "    \"selectedHoles\": " << ops.selected_holes << ",\n";
  out << "    \"filledHoles\": " << ops.filled_holes << ",\n";
  out << "    \"skippedByEdgeLimit\": " << hole_report.skipped_by_edge_limit << ",\n";
  out << "    \"skippedByDiameterLimit\": " << hole_report.skipped_by_diameter_limit << ",\n";
  out << "    \"boundaryEdgeCountQuantiles\": ";
  write_json_number_array(out, quantiles(hole_edges));
  out << ",\n";
  out << "    \"bboxDiagonalQuantiles\": ";
  write_json_number_array(out, quantiles(hole_diameters));
  out << "\n";
  out << "  },\n";
  out << "  \"beforeStats\": {\n";
  write_stats(out, before, "    ");
  out << "  },\n";
  out << "  \"afterStats\": {\n";
  write_stats(out, after, "    ");
  out << "  }\n";
  out << "}\n";
}

int main(int argc, char** argv) {
  try {
    Options opts = parse_args(argc, argv);
    Mesh mesh;
    if (!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(
            opts.input_mesh,
            mesh,
            CGAL::parameters::repair_polygon_soup(false).verbose(true)) ||
        num_faces(mesh) == 0) {
      throw std::runtime_error("Could not read a non-empty polygon mesh: " + opts.input_mesh);
    }
    if (!CGAL::is_triangle_mesh(mesh)) {
      throw std::runtime_error("Expected a triangular mesh: " + opts.input_mesh);
    }

    const MeshStats before = collect_stats(mesh, opts.detect_self_intersections);
    Operations ops;
    HoleReport hole_report;
    Mesh filled_patch_mesh;

    if (opts.repair_degenerate_faces) {
      ops.degenerate_faces_repair_ok = PMP::remove_degenerate_faces(mesh);
      PMP::remove_isolated_vertices(mesh);
      mesh.collect_garbage();
    }
    if (opts.repair_almost_degenerate_faces) {
      ops.almost_degenerate_faces_repair_ok = PMP::remove_almost_degenerate_faces(mesh);
      PMP::remove_isolated_vertices(mesh);
      mesh.collect_garbage();
    }
    if (opts.duplicate_nonmanifold_vertices) {
      ops.duplicated_nonmanifold_vertices = PMP::duplicate_non_manifold_vertices(mesh);
      mesh.collect_garbage();
    }
    if (opts.stitch_borders) {
      ops.stitched_halfedge_pairs = PMP::stitch_borders(mesh);
      mesh.collect_garbage();
    }
    if (opts.fill_holes) {
      fill_selected_holes(mesh, opts, ops, hole_report, filled_patch_mesh);
      PMP::remove_isolated_vertices(mesh);
      mesh.collect_garbage();
    }
    if (opts.component_area_threshold > 0.0) {
      ops.removed_tiny_components = PMP::remove_connected_components_of_negligible_size(
          mesh,
          CGAL::parameters::area_threshold(opts.component_area_threshold).volume_threshold(0.0));
      PMP::remove_isolated_vertices(mesh);
      mesh.collect_garbage();
    }
    if (opts.repair_self_intersections) {
      try {
        ops.self_intersection_repair_ok = PMP::experimental::remove_self_intersections(mesh);
        PMP::remove_isolated_vertices(mesh);
        mesh.collect_garbage();
      } catch (const std::exception& exc) {
        ops.self_intersection_repair_ok = false;
        ops.self_intersection_repair_error = exc.what();
      }
    }

    const MeshStats after = collect_stats(mesh, opts.detect_self_intersections);
    if (!CGAL::IO::write_polygon_mesh(opts.output_mesh, mesh, CGAL::parameters::stream_precision(17))) {
      throw std::runtime_error("Could not write polygon mesh: " + opts.output_mesh);
    }
    if (!opts.filled_patch_mesh.empty() && num_faces(filled_patch_mesh) > 0) {
      if (!CGAL::IO::write_polygon_mesh(opts.filled_patch_mesh, filled_patch_mesh, CGAL::parameters::stream_precision(17))) {
        throw std::runtime_error("Could not write filled patch mesh: " + opts.filled_patch_mesh);
      }
    }
    write_hole_report_jsonl(opts, hole_report);
    write_summary(opts, before, after, ops, hole_report);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "cgal_mesh_heal error: " << exc.what() << "\n";
    return 1;
  }
}
