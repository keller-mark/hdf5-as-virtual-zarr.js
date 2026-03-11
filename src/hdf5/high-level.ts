/**
 * Vendored and adapted from jsfive/esm/high-level.js.
 * File, Group, Dataset — the public-facing HDF5 object model.
 *
 * All classes that read from the file use async factory methods,
 * delegating byte-range reads to Source.
 */

import type { Source } from "../index.js";
import { DataObjects } from "./dataobjects.js";
import { SuperBlock } from "./misc-low-level.js";

// ────────── Group ──────────

export class Group {
  parent: Group | File;
  file: File;
  name: string;
  _links: Record<string, any>;
  _dataobjects: DataObjects;
  private _attrs: Record<string, any> | null = null;
  private _keys: string[] | null = null;

  constructor(
    name: string,
    dataobjects: DataObjects,
    links: Record<string, any>,
    parent?: Group | File
  ) {
    if (parent == null) {
      this.parent = this as any;
      this.file = this as any as File;
    } else {
      this.parent = parent;
      this.file = parent instanceof File ? parent : parent.file;
    }
    this.name = name;
    this._links = links;
    this._dataobjects = dataobjects;
  }

  get keys(): string[] {
    if (this._keys == null) {
      this._keys = Object.keys(this._links);
    }
    return this._keys.slice();
  }

  async get(y: string): Promise<Group | Dataset> {
    let path = normpath(y);
    if (path === "/") return this.file;
    if (path === ".") return this;
    if (/^\//.test(path)) return this.file.get(path.slice(1));

    let next_obj: string;
    let additional_obj: string;
    if (posix_dirname(path) !== "") {
      [next_obj, additional_obj] = path.split(/\/(.*)/);
    } else {
      next_obj = path;
      additional_obj = ".";
    }

    if (!(next_obj in this._links)) {
      throw new Error(next_obj + " not found in group");
    }

    const obj_name = normpath(this.name + "/" + next_obj);
    const link_target = this._links[next_obj];

    if (typeof link_target === "string") {
      try {
        return await this.get(link_target);
      } catch {
        throw new Error("Broken soft link: " + link_target);
      }
    }

    const source = this.file._source;
    const dataobjs = await DataObjects.create(source, link_target);

    if (dataobjs.is_dataset) {
      if (additional_obj !== ".") {
        throw new Error(obj_name + " is a dataset, not a group");
      }
      return new Dataset(obj_name, dataobjs, this);
    } else {
      const links = await dataobjs.get_links();
      const new_group = new Group(obj_name, dataobjs, links, this);
      return new_group.get(additional_obj);
    }
  }

  async get_attrs(): Promise<Record<string, any>> {
    if (this._attrs == null) {
      this._attrs = await this._dataobjects.get_attributes();
    }
    return this._attrs;
  }
}

// ────────── File ──────────

export class File extends Group {
  _source: Source;
  filename: string;
  mode = "r";
  userblock_size = 0;

  private constructor(
    source: Source,
    dataobjects: DataObjects,
    links: Record<string, any>,
    filename: string
  ) {
    super("/", dataobjects, links);
    this.parent = this;
    this.file = this;
    this._source = source;
    this.filename = filename;
  }

  static async create(source: Source, filename?: string): Promise<File> {
    const superblock = await SuperBlock.create(source, 0);
    const offset = await superblock.getOffsetToDataobjects();
    const dataobjects = await DataObjects.create(source, offset);
    const links = await dataobjects.get_links();
    return new File(source, dataobjects, links, filename || "");
  }
}

// ────────── Dataset ──────────

export class Dataset {
  parent: Group | File;
  file: File;
  name: string;
  _dataobjects: DataObjects;

  constructor(name: string, dataobjects: DataObjects, parent: Group | File) {
    this.parent = parent;
    this.file = parent instanceof File ? parent : parent.file;
    this.name = name;
    this._dataobjects = dataobjects;
  }

  get shape(): number[] {
    return this._dataobjects.shape;
  }

  async get_attrs(): Promise<Record<string, any>> {
    return this._dataobjects.get_attributes();
  }

  get dtype(): any {
    return this._dataobjects.dtype;
  }

  get fillvalue(): any {
    return this._dataobjects.fillvalue;
  }

  async get_data(): Promise<any[]> {
    return this._dataobjects.get_data();
  }
}

// ────────── path helpers ──────────

function posix_dirname(p: string): string {
  const sep = "/";
  const i = p.lastIndexOf(sep) + 1;
  let head = p.slice(0, i);
  if (head && !/^\/+$/.test(head)) {
    head = head.replace(/\/$/, "");
  }
  return head;
}

function normpath(path: string): string {
  return path.replace(/\/(\/)+/g, "/");
}
