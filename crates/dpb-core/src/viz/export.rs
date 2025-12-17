//! Export utilities for SVG and JSON generation.

use super::PlotConfig;

/// Builder for creating SVG documents.
#[derive(Debug, Clone)]
pub struct SvgBuilder {
    width: u32,
    height: u32,
    elements: Vec<String>,
}

impl SvgBuilder {
    /// Creates a new SVG builder with specified dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            elements: Vec::new(),
        }
    }

    /// Adds a line element.
    pub fn line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, stroke: &str, width: f32) -> &mut Self {
        self.elements.push(format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" />"#,
            x1, y1, x2, y2, stroke, width
        ));
        self
    }

    /// Adds a rectangle element.
    pub fn rect(&mut self, x: f32, y: f32, width: f32, height: f32, fill: &str) -> &mut Self {
        self.elements.push(format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" />"#,
            x, y, width, height, fill
        ));
        self
    }

    /// Adds a rectangle with stroke.
    pub fn rect_stroke(&mut self, x: f32, y: f32, width: f32, height: f32, fill: &str, stroke: &str, stroke_width: f32) -> &mut Self {
        self.elements.push(format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="{}" />"#,
            x, y, width, height, fill, stroke, stroke_width
        ));
        self
    }

    /// Adds a circle element.
    pub fn circle(&mut self, cx: f32, cy: f32, r: f32, fill: &str) -> &mut Self {
        self.elements.push(format!(
            r#"<circle cx="{}" cy="{}" r="{}" fill="{}" />"#,
            cx, cy, r, fill
        ));
        self
    }

    /// Adds a text element.
    pub fn text(&mut self, x: f32, y: f32, content: &str, font_size: u32, fill: &str) -> &mut Self {
        self.elements.push(format!(
            r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="{}" fill="{}">{}</text>"#,
            x, y, font_size, fill, Self::escape_xml(content)
        ));
        self
    }

    /// Adds a text element with anchor.
    pub fn text_anchor(&mut self, x: f32, y: f32, content: &str, font_size: u32, fill: &str, anchor: &str) -> &mut Self {
        self.elements.push(format!(
            r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="{}" fill="{}" text-anchor="{}">{}</text>"#,
            x, y, font_size, fill, anchor, Self::escape_xml(content)
        ));
        self
    }

    /// Adds a path element.
    pub fn path(&mut self, d: &str, stroke: &str, width: f32, fill: &str) -> &mut Self {
        self.elements.push(format!(
            r#"<path d="{}" stroke="{}" stroke-width="{}" fill="{}" />"#,
            d, stroke, width, fill
        ));
        self
    }

    /// Adds a polyline element.
    pub fn polyline(&mut self, points: &[(f32, f32)], stroke: &str, width: f32, fill: &str) -> &mut Self {
        let points_str = points
            .iter()
            .map(|(x, y)| format!("{},{}", x, y))
            .collect::<Vec<_>>()
            .join(" ");
        self.elements.push(format!(
            r#"<polyline points="{}" stroke="{}" stroke-width="{}" fill="{}" />"#,
            points_str, stroke, width, fill
        ));
        self
    }

    /// Adds axes to the plot.
    pub fn axes(&mut self, margin: f32, config: &PlotConfig) -> &mut Self {
        let width = config.width as f32;
        let height = config.height as f32;

        // X axis
        self.line(margin, height - margin, width - margin, height - margin, "#000000", 2.0);
        // Y axis
        self.line(margin, margin, margin, height - margin, "#000000", 2.0);

        // Add axis labels if present
        if let Some(ref x_label) = config.x_label {
            self.text_anchor(width / 2.0, height - 10.0, x_label, 14, "#000000", "middle");
        }
        if let Some(ref y_label) = config.y_label {
            // Rotated text would require transform, for simplicity we'll place it horizontally
            self.text_anchor(10.0, height / 2.0, y_label, 14, "#000000", "middle");
        }

        self
    }

    /// Adds a grid to the plot.
    pub fn grid(&mut self, margin: f32, width: u32, height: u32, num_x: usize, num_y: usize) -> &mut Self {
        let w = width as f32;
        let h = height as f32;
        let plot_width = w - 2.0 * margin;
        let plot_height = h - 2.0 * margin;

        // Vertical grid lines
        for i in 0..=num_x {
            let x = margin + (i as f32 / num_x as f32) * plot_width;
            self.line(x, margin, x, h - margin, "#e0e0e0", 1.0);
        }

        // Horizontal grid lines
        for i in 0..=num_y {
            let y = margin + (i as f32 / num_y as f32) * plot_height;
            self.line(margin, y, w - margin, y, "#e0e0e0", 1.0);
        }

        self
    }

    /// Adds a title to the plot.
    pub fn title(&mut self, title: &str, width: u32) -> &mut Self {
        self.text_anchor(width as f32 / 2.0, 25.0, title, 18, "#000000", "middle");
        self
    }

    /// Builds the final SVG string.
    pub fn build(&self) -> String {
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        );
        svg.push('\n');

        // Background
        svg.push_str(&format!(
            r##"  <rect width="{}" height="{}" fill="#ffffff" />"##,
            self.width, self.height
        ));
        svg.push('\n');

        // Add all elements
        for elem in &self.elements {
            svg.push_str("  ");
            svg.push_str(elem);
            svg.push('\n');
        }

        svg.push_str("</svg>");
        svg
    }

    /// Escapes XML special characters.
    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }
}

/// Builder for creating JSON documents.
#[derive(Debug, Clone)]
pub struct JsonBuilder {
    fields: Vec<(String, String)>,
}

impl JsonBuilder {
    /// Creates a new JSON builder.
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
        }
    }

    /// Adds a string field.
    pub fn add_string(&mut self, key: &str, value: &str) -> &mut Self {
        self.fields.push((key.to_string(), format!(r#""{}""#, Self::escape_json(value))));
        self
    }

    /// Adds a number field.
    pub fn add_number(&mut self, key: &str, value: f64) -> &mut Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }

    /// Adds an integer field.
    pub fn add_int(&mut self, key: &str, value: i64) -> &mut Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }

    /// Adds a boolean field.
    pub fn add_bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }

    /// Adds an array of numbers.
    pub fn add_number_array(&mut self, key: &str, values: &[f64]) -> &mut Self {
        let array = format!(
            "[{}]",
            values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
        );
        self.fields.push((key.to_string(), array));
        self
    }

    /// Adds a 2D array of numbers.
    pub fn add_number_array_2d(&mut self, key: &str, values: &[Vec<f64>]) -> &mut Self {
        let array = format!(
            "[{}]",
            values
                .iter()
                .map(|row| format!("[{}]", row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")))
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.fields.push((key.to_string(), array));
        self
    }

    /// Adds an array of strings.
    pub fn add_string_array(&mut self, key: &str, values: &[String]) -> &mut Self {
        let array = format!(
            "[{}]",
            values.iter().map(|v| format!(r#""{}""#, Self::escape_json(v))).collect::<Vec<_>>().join(", ")
        );
        self.fields.push((key.to_string(), array));
        self
    }

    /// Adds raw JSON value.
    pub fn add_raw(&mut self, key: &str, value: &str) -> &mut Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }

    /// Builds the final JSON string.
    pub fn build(&self) -> String {
        let mut json = String::from("{\n");

        for (i, (key, value)) in self.fields.iter().enumerate() {
            json.push_str(&format!(r#"  "{}": {}"#, Self::escape_json(key), value));
            if i < self.fields.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }

        json.push('}');
        json
    }

    /// Escapes JSON special characters.
    fn escape_json(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }
}

impl Default for JsonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svg_builder_line() {
        let mut svg = SvgBuilder::new(100, 100);
        svg.line(10.0, 10.0, 90.0, 90.0, "#000000", 2.0);
        let result = svg.build();
        assert!(result.contains(r#"<line x1="10" y1="10" x2="90" y2="90""#));
    }

    #[test]
    fn test_svg_builder_rect() {
        let mut svg = SvgBuilder::new(100, 100);
        svg.rect(10.0, 10.0, 50.0, 50.0, "#ff0000");
        let result = svg.build();
        assert!(result.contains(r##"<rect x="10" y="10" width="50" height="50" fill="#ff0000""##));
    }

    #[test]
    fn test_svg_builder_text() {
        let mut svg = SvgBuilder::new(100, 100);
        svg.text(50.0, 50.0, "Hello", 12, "#000000");
        let result = svg.build();
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_json_builder_string() {
        let mut json = JsonBuilder::new();
        json.add_string("name", "test");
        let result = json.build();
        assert!(result.contains(r#""name": "test""#));
    }

    #[test]
    fn test_json_builder_number() {
        let mut json = JsonBuilder::new();
        json.add_number("value", 42.5);
        let result = json.build();
        assert!(result.contains(r#""value": 42.5"#));
    }

    #[test]
    fn test_json_builder_array() {
        let mut json = JsonBuilder::new();
        json.add_number_array("values", &[1.0, 2.0, 3.0]);
        let result = json.build();
        assert!(result.contains(r#""values": [1, 2, 3]"#));
    }
}
